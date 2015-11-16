
object Cells {
  :cp /Users/g09971710/src/wordpress-workshop/target/scala-2.10/wordpress-workshop-assembly-0.1.0-SNAPSHOT.jar

  /* ... new cell ... */

  import org.apache.spark.SparkContext
  import org.apache.spark.broadcast.Broadcast
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.param.ParamMap
  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  import org.apache.spark.mllib.linalg.{Vector, Vectors}
  import org.apache.spark.rdd.RDD
  import org.apache.spark.sql.{DataFrame, Row}
  import scalaz.Scalaz._

  /* ... new cell ... */

  import wordpressworkshop._

  /* ... new cell ... */

  @transient val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  /* ... new cell ... */

  val dataPath = "file:///Users/g09971710/src/wordpress-workshop/data/"

  /* ... new cell ... */

  val statsBlogDF = sqlContext.read.json(dataPath + "kaggle-stats-blogs-20111123-20120423.json")
  val statsBlogRDD = statsBlogDF.map(_.toSeq.toList match {
    case List(blogId: Long, numLikes: Long, numPosts: Long) => StatsBlog(blogId, numLikes, numPosts)
  })

  /* ... new cell ... */

  val statsUserDF = sqlContext.read.json(dataPath + "kaggle-stats-users-20111123-20120423.json")
  val statsUserRDD: RDD[StatsUser] = statsUserDF.map(_.toSeq.toList match {
    case List(likeBlogs: Seq[Row], numLikes: Long, userId: Long) =>
      StatsUser(userId, numLikes, likeBlogs.map(row => row.getAs[String](0).toLong -> row.getAs[String](1).toLong).toMap)
  })

  /* ... new cell ... */

  val testUsers = sc.textFile(dataPath + "test.csv").filter(!_.startsWith("id")).map(_.toLong)

  /* ... new cell ... */

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

  /* ... new cell ... */

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

  /* ... new cell ... */

  val evaluationTest: RDD[(Long, List[Long])] =
        sc.textFile(dataPath + "evaluation-final.csv").map(_.split(",").toList match {
          case List(userId, posts, _) => userId.toLong -> posts.split(" ").toList.map(_.toLong)
        })

  /* ... new cell ... */

  statsUserRDD.filter(statsUser => statsUser.likeBlogs.values.sum != statsUser.numLikes).count()

  /* ... new cell ... */

  statsUserRDD.map(_.numLikes).histogram(25) match {
    case (bins, counts) => bins.zip(counts)
  }

  /* ... new cell ... */

  val bins = (0 to 24).map(_.toDouble).toArray
  
  bins.zip(statsUserRDD.map(_.numLikes).histogram(bins))

  /* ... new cell ... */

  val categories = trainPostsRDD.flatMap(_._1.categories).map(_ -> 1)
  .reduceByKey(_ + _).sortBy(_._2, ascending = false).take(25).toList

  /* ... new cell ... */

  trainPostsRDD.flatMap(_._2).distinct()
  .map(_ -> ()).leftOuterJoin(statsUserRDD.keyBy(_.userId)).filter(_._2._2.isEmpty).count()

  /* ... new cell ... */

  statsUserRDD.map(_.userId).distinct().count()

  /* ... new cell ... */

  trainPostsRDD.flatMap(_._2).distinct().count()

  /* ... new cell ... */

  // missing
  trainPostsRDD.map(_._1.blogId).distinct()
  .map(_ -> ()).leftOuterJoin(statsBlogRDD.keyBy(_.blogId)).filter(_._2._2.isEmpty).count()

  /* ... new cell ... */

  // defined
  trainPostsRDD.map(_._1.blogId).distinct()
  .map(_ -> ()).leftOuterJoin(statsBlogRDD.keyBy(_.blogId)).filter(_._2._2.isDefined).count()

  /* ... new cell ... */

  trainPostsRDD.map(_._1.blogId).distinct().count()

  /* ... new cell ... */

  statsBlogRDD.count()

  /* ... new cell ... */

  trainPostsRDD.filter(_._1.language.isEmpty).count()

  /* ... new cell ... */

  trainPostsRDD.keys.filter(post => post.blogId == 4 && post.postId == 1182450).collect().toList

  /* ... new cell ... */

  trainPostsRDD.flatMap(_.productIterator).filter(_ == null).count()

  /* ... new cell ... */

  val blogIdToPriorBlogLikelihoodBV: Broadcast[Map[Long, Double]] = 
    sc.broadcast(Features.blogIdToPriorBlogLikelihoodBV(statsUserRDD))

  /* ... new cell ... */

  val meanBlogLikesPerPost: Double = Features.meanBlogLikesPerPost(statsBlogRDD)

  /* ... new cell ... */

  type OtherLikelihoodMaps = (Map[String, Int], Map[String, Int], Map[String, Int], Map[Long, Int], Map[Int, Int])
  val userIdToOtherLikelihoodMaps: RDD[(Long, OtherLikelihoodMaps)] = Features.userIdToOtherLikelihoodMaps(trainPostsRDD)
  val userIdToOtherLikelihoodMapsBV: Broadcast[Map[Long, OtherLikelihoodMaps]] =
    sc.broadcast(userIdToOtherLikelihoodMaps.collect().toMap)

  /* ... new cell ... */

  val blogIdToAverageLikesPerPost: Map[Long, Double] = statsBlogRDD.map {
    case StatsBlog(blogId, numLikes, numPosts) => blogId -> numLikes.toDouble / numPosts
  }.collect().toMap
  val blogIdToAverageLikesPerPostBV: Broadcast[Map[Long, Double]] = sc.broadcast(blogIdToAverageLikesPerPost)

  /* ... new cell ... */

  val userIdToBlogLikelihood: Map[Long, Map[Long, Double]] =
    statsUserRDD.map {
      case StatsUser(userId, numLikes, likeBlogs) => userId -> likeBlogs.mapValues(_.toDouble / numLikes).map(identity)
    }
    .filter(_._2.size < 1000).collect().toMap
  val userIdToBlogLikelihoodBV: Broadcast[Map[Long, Map[Long, Double]]] = sc.broadcast(userIdToBlogLikelihood)

  /* ... new cell ... */

  val featuresOfTrue = Features.features(trainPostsRDD, userIdToOtherLikelihoodMapsBV,
      userIdToBlogLikelihoodBV, blogIdToPriorBlogLikelihoodBV, blogIdToAverageLikesPerPostBV, meanBlogLikesPerPost)

  /* ... new cell ... */

  val numLikeUsersDistributionArrayBV: Broadcast[Array[Int]] =
        sc.broadcast(FalseLikes.numLikeUsersDistributionArray(trainPostsRDD))

  /* ... new cell ... */

  val userIds: Broadcast[Set[Long]] = sc.broadcast(trainPostsRDD.flatMap(_._2).distinct().collect().toSet)

  /* ... new cell ... */

  val blogPostsWithNonLikeUsers: RDD[(BlogPost, Set[Long])] =
        FalseLikes.blogPostsWithNonLikeUsers(trainPostsRDD, numLikeUsersDistributionArrayBV, userIds)

  /* ... new cell ... */

  val featuresOfFalse = Features.features(blogPostsWithNonLikeUsers, userIdToOtherLikelihoodMapsBV,
        userIdToBlogLikelihoodBV, blogIdToPriorBlogLikelihoodBV, blogIdToAverageLikesPerPostBV, meanBlogLikesPerPost)

  /* ... new cell ... */

  val features = featuresOfTrue.values.map(features => 1.0 ->
    Vectors.dense(features.productIterator.toArray.map(_.asInstanceOf[Double]))
  ) ++ featuresOfFalse.values.map(features => 0.0 ->
    Vectors.dense(features.productIterator.toArray.map(_.asInstanceOf[Double]))
  )

  /* ... new cell ... */

  // persist training features
  features.map {
    case (label, vector) => (label +: vector.toArray.toList).mkString("\t")
  }
  .saveAsTextFile(dataPath + "trainFeatures-final2")

  /* ... new cell ... */

  val postTestIds: Set[Long] = evaluationTest.flatMap(_._2).distinct().collect().toSet
  val postTestIdsBV: Broadcast[Set[Long]] = sc.broadcast(postTestIds)

  /* ... new cell ... */

  val testUsers: RDD[Long] = sc.textFile(dataPath + "test.csv").filter(!_.startsWith("id")).map(_.toLong)
  val smallTestUsers: Broadcast[Set[Long]] =
    sc.broadcast(testUsers.sample(withReplacement = false, fraction = 0.1).take(100).toSet)

  /* ... new cell ... */

  val testUserSetBV = sc.broadcast(testUsers.collect().toSet)

  /* ... new cell ... */

  val smallTestUsers: Broadcast[Set[Long]] =
        sc.broadcast(testUsers.sample(withReplacement = false, fraction = 0.1).take(100).toSet)

  /* ... new cell ... */

  val tagToUserIds =
    userIdToOtherLikelihoodMaps.filter(_._1 |> testUserSetBV.value).flatMap {
      case (userId, maps) => maps._2.keys.map(_ -> userId)
    }
    .distinct().groupByKey().mapValues(_.toSet).collect().toMap
  val tagToUserIdsBV = sc.broadcast(tagToUserIds)

  /* ... new cell ... */

  val testBlogPostsWithTestUsers: RDD[(BlogPost, Set[Long])] =
    testPostsRDD.filter(_.postId |> postTestIdsBV.value)
    .map(post => post -> post.tags.flatMap(tagToUserIdsBV.value.get).flatten)
    .filter(_._2.nonEmpty)

  /* ... new cell ... */

  val testFeatures =
    Features.features(testBlogPostsWithTestUsers, userIdToOtherLikelihoodMapsBV,
      userIdToBlogLikelihoodBV, blogIdToPriorBlogLikelihoodBV, blogIdToAverageLikesPerPostBV, meanBlogLikesPerPost)

  /* ... new cell ... */

  testFeatures.map{
    case ((userId, postId), features) => (List(userId, postId) ++ features.productIterator).mkString("\t")
  }
  .saveAsTextFile(dataPath + "testFeatures-final-complete")

  /* ... new cell ... */

  sc.textFile(dataPath + "testFeatures-final").take(10)
}
              