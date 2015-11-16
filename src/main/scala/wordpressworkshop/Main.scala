package wordpressworkshop

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

import scalaz.Scalaz._

object Main {
  @transient val sc = new SparkContext()
  @transient val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import sqlContext.implicits._

  val dataPath = "file:///Users/g09971710/src/wordpress-workshop/data/"
  def featureExtraction = {
    val statsBlogDF = sqlContext.read.json(dataPath + "kaggle-stats-blogs-20111123-20120423.json")
    val statsBlogRDD = statsBlogDF.map(_.toSeq.toList match {
      case List(blogId: Long, numLikes: Long, numPosts: Long) => StatsBlog(blogId, numLikes, numPosts)
    })

    val statsUserDF = sqlContext.read.json(dataPath + "kaggle-stats-users-20111123-20120423.json")
    val statsUserRDD: RDD[StatsUser] = statsUserDF.map(_.toSeq.toList match {
      case List(likeBlogs: Seq[Row], numLikes: Long, userId: Long) =>
        StatsUser(userId, numLikes, likeBlogs.map(row => row.getAs[String](0).toLong -> row.getAs[String](1).toLong).toMap)
    })

    val trainPostsDF = sqlContext.read.json(dataPath + "final-data-non-thin/trainPosts.json")
    val trainPostsRDD: RDD[(BlogPost, Set[Long])] = trainPostsDF.na.drop().map(_.toSeq.toList match {
      case List(authorId: String, blogId: String, blogName: String, categories: Seq[String], content,
      _, _, language: String, likes: Seq[Row], postId: String,
      tags: Seq[String], title, url: String) =>
        BlogPost(blogId.toLong, postId.toLong, authorId.toLong, blogName, Option(title.asInstanceOf[String]),
          Option(content.asInstanceOf[String]),
          language, categories.toSet,
          tags.toSet) -> likes.map(_.getAs[String](1).toLong).toSet
    })

    val testPostsDF = sqlContext.read.json(dataPath + "final-data-non-thin/testPosts.json")
    val testPostsRDD: RDD[BlogPost] = testPostsDF.na.drop.map(_.toSeq.toList match {
      case List(authorId: String, blogId: String, blogName: String, categories: Seq[String], content,
      _, _, language: String, likes: Seq[Row], postId: String,
      tags: Seq[String], title, url: String) =>
        BlogPost(blogId.toLong, postId.toLong, authorId.toLong, blogName, Option(title.asInstanceOf[String]),
          Option(content.asInstanceOf[String]),
          language, categories.toSet,
          tags.toSet)
    })

    val blogIdToPriorBlogLikelihoodBV: Broadcast[Map[Long, Double]] =
      sc.broadcast(Features.blogIdToPriorBlogLikelihoodBV(statsUserRDD))

    val meanBlogLikesPerPost: Double = Features.meanBlogLikesPerPost(statsBlogRDD)

    type OtherLikelihoodMaps = (Map[String, Int], Map[String, Int], Map[String, Int], Map[Long, Int], Map[Int, Int])
    val userIdToOtherLikelihoodMaps: RDD[(Long, OtherLikelihoodMaps)] = Features.userIdToOtherLikelihoodMaps(trainPostsRDD)
    val userIdToOtherLikelihoodMapsBV: Broadcast[Map[Long, OtherLikelihoodMaps]] =
      sc.broadcast(userIdToOtherLikelihoodMaps.collect().toMap)

    val blogIdToAverageLikesPerPost: Map[Long, Double] = statsBlogRDD.map {
      case StatsBlog(blogId, numLikes, numPosts) => blogId -> numLikes.toDouble / numPosts
    }.collect().toMap
    val blogIdToAverageLikesPerPostBV: Broadcast[Map[Long, Double]] = sc.broadcast(blogIdToAverageLikesPerPost)

    val userIdToBlogLikelihood: Map[Long, Map[Long, Double]] =
      statsUserRDD.map {
        case StatsUser(userId, numLikes, likeBlogs) => userId -> likeBlogs.mapValues(_.toDouble / numLikes).map(identity)
      }
      .filter(_._2.size < 1000).collect().toMap
    val userIdToBlogLikelihoodBV: Broadcast[Map[Long, Map[Long, Double]]] = sc.broadcast(userIdToBlogLikelihood)

    val featuresOfTrue = Features.features(trainPostsRDD, userIdToOtherLikelihoodMapsBV,
      userIdToBlogLikelihoodBV, blogIdToPriorBlogLikelihoodBV, blogIdToAverageLikesPerPostBV, meanBlogLikesPerPost)

    val numLikeUsersDistributionArrayBV: Broadcast[Array[Int]] =
      sc.broadcast(FalseLikes.numLikeUsersDistributionArray(trainPostsRDD))

    val userIds: Broadcast[Set[Long]] = sc.broadcast(trainPostsRDD.flatMap(_._2).distinct().collect().toSet)

    val blogPostsWithNonLikeUsers: RDD[(BlogPost, Set[Long])] =
      FalseLikes.blogPostsWithNonLikeUsers(trainPostsRDD, numLikeUsersDistributionArrayBV, userIds)

    val featuresOfFalse = Features.features(blogPostsWithNonLikeUsers, userIdToOtherLikelihoodMapsBV,
      userIdToBlogLikelihoodBV, blogIdToPriorBlogLikelihoodBV, blogIdToAverageLikesPerPostBV, meanBlogLikesPerPost)

    val features = featuresOfTrue.values.map(features => 1.0 ->
      Vectors.dense(features.productIterator.toArray.map(_.asInstanceOf[Double]))
    ) ++ featuresOfFalse.values.map(features => 0.0 ->
      Vectors.dense(features.productIterator.toArray.map(_.asInstanceOf[Double]))
    )

    // persist training features
    features.map {
      case (label, vector) => (label +: vector.toArray.toList).mkString("\t")
    }
    .saveAsTextFile(dataPath + "trainFeatures-final")

    // EVALUATION TEST DATA
    val evaluationTest: RDD[(Long, List[Long])] =
      sc.textFile(dataPath + "evaluation-final.csv").map(_.split(",").toList match {
        case List(userId, posts, _) => userId.toLong -> posts.split(" ").toList.map(_.toLong)
      })

    val postTestIds: Set[Long] = evaluationTest.flatMap(_._2).distinct().collect().toSet
    val postTestIdsBV: Broadcast[Set[Long]] = sc.broadcast(postTestIds)

    val testUsers: RDD[Long] = sc.textFile(dataPath + "test.csv").filter(!_.startsWith("id")).map(_.toLong)
    val smallTestUsers: Broadcast[Set[Long]] =
      sc.broadcast(testUsers.sample(withReplacement = false, fraction = 0.1).take(100).toSet)

    //val testUserSetBV = sc.broadcast(smallTestUsers.collect().toSet)

    val tagToUserIds =
      userIdToOtherLikelihoodMaps.filter(_._1 |> smallTestUsers.value).flatMap {
        case (userId, maps) => maps._2.keys.map(_ -> userId)
      }
      .distinct().groupByKey().mapValues(_.toSet).collect().toMap
    val tagToUserIdsBV = sc.broadcast(tagToUserIds)

    val testBlogPostsWithTestUsers: RDD[(BlogPost, Set[Long])] =
      testPostsRDD.filter(_.postId |> postTestIdsBV.value)
      .map(post => post -> post.tags.flatMap(tagToUserIdsBV.value.get).flatten)
      .filter(_._2.nonEmpty)

    val testFeatures: RDD[((Long, Long), Features)] =
      Features.features(testBlogPostsWithTestUsers, userIdToOtherLikelihoodMapsBV,
        userIdToBlogLikelihoodBV, blogIdToPriorBlogLikelihoodBV, blogIdToAverageLikesPerPostBV, meanBlogLikesPerPost)

    testFeatures.values.map(features => 1.0 ->
      Vectors.dense(features.productIterator.toArray.map(_.asInstanceOf[Double]))
    ).map {
      case (label, vector) => (label +: vector.toArray.toList).mkString("\t")
    }
    .saveAsTextFile(dataPath + "testFeatures-final")
  }

  def classifierEvaluationTrainingOnly {
    val features =
      sc.textFile(dataPath + "trainFeatures-final").map(_.split("\t").map(_.toDouble).toList match {
        case label :: vector => label -> Vectors.dense(vector.toArray)
      })

    val validFeatures: RDD[(Double, Vector)] = features.filter(!_._2.toArray.exists(x => x.isInfinity || x.isNaN))

    val training: DataFrame = validFeatures.toDF("label", "features")

    val lr = new LogisticRegression()
    // Print out the parameters, documentation, and any default values.
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    val paramMap = ParamMap(lr.maxIter -> 20)
                   .put(lr.regParam -> 0.01)
                   .put(lr.probabilityCol -> "probability")

    val Array(trainingData, testData) = training.randomSplit(Array(0.8, 0.2))

    val model = lr.fit(trainingData, paramMap)

    val predictions = model.transform(testData)
    val predictionAndLabels = predictions.map(row => row.getAs[Vector](3)(1) -> row.getAs[Double](0))

    import Pimps._

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    metrics.precisionByThreshold().collect().interpolateLinear(25).roundX(2)

    // Precision by threshold
    metrics.precisionByThreshold().collect().interpolatePercentiles(25).roundX(2)

    // Recall by threshold
    metrics.recallByThreshold().collect().interpolatePercentiles(25).roundX(2)

    // Precision-Recall Curve = recall = f(precision)
    metrics.pr().collect().interpolatePercentiles(25).roundX(4)

    // F-measure
    metrics.fMeasureByThreshold().collect().interpolatePercentiles(25).roundX(3)
    val beta = 0.5
    metrics.fMeasureByThreshold(beta).collect().interpolatePercentiles(25).roundX(3)

    // AUPRC
    metrics.areaUnderPR()

    // Compute thresholds used in ROC and PR curves
    val thresholds = metrics.precisionByThreshold().map(_._1)
    val thresholdIndexes = thresholds.collect().zipWithIndex

    // ROC Curve
    metrics.roc.collect().interpolateLinear(25)

    // AUROC
    val auROC = metrics.areaUnderROC
  }

  def recommendationEvaluation = {
    val trainingRDD: RDD[(Double, Vector)] =
      sc.textFile(dataPath + "trainFeatures-final").map(_.split("\t").map(_.toDouble).toList match {
        case label :: vector => label -> Vectors.dense(vector.toArray)
      })

    val training: DataFrame =
      trainingRDD.filter(!_._2.toArray.exists(x => x.isInfinity || x.isNaN)).toDF("label", "features")

    val testRDD: RDD[((Long, Long), Vector)] =
      sc.textFile(dataPath + "testFeatures-final").map(_.split("\t").toList match {
        case userId :: postId :: vector =>
          (userId.toLong, postId.toLong) -> Vectors.dense(vector.map(_.toDouble).toArray)
      })

    val evaluationTest: RDD[(Long, Set[Long])] =
      sc.textFile(dataPath + "evaluation-final.csv").map(_.split(",").toList match {
        case List(userId, posts, _) => userId.toLong -> posts.split(" ").map(_.toLong).toSet
      })

    val testUserIdToPostLikesBV: Broadcast[Map[Long, Set[Long]]] = sc.broadcast(evaluationTest.collect().toMap)

    val testData: DataFrame = testRDD.map {
      case ((userId, postId), vector) => (userId, postId,
        if (testUserIdToPostLikesBV.value(userId)(postId)) 1.0 else 0.0, vector)
    }.toDF("userId", "postId", "label", "features")

    val lrr = LogisticRegressionRecommender(training)
    lrr.model.weights
    //    @transient val metrics = lrr.metrics(testData)
    //    val thresholds = metrics.thresholds()

    //    val bestThresholdFromROC = 1 - thresholds.collect().zip(metrics.roc().collect()).filter(_._2 |> {
    //      case (fpr, tpr) => tpr > 0.80
    //    }).minBy(_._2._1)._1

    val lrrMAPScore =
      MAP.tuneThreshold(50, 100, 2)(lrr.likeScores(testData), evaluationTest).maxBy(_._2)

    val tagMAPScore = MAP.tuneThreshold(1, 100, 1)(TagLikelihoodRecommender.likeScores(testData), evaluationTest)

  }

}
