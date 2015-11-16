
object Cells {
  :cp /Users/g09971710/src/wordpress-workshop/target/scala-2.10/wordpress-workshop-assembly-0.1.0-SNAPSHOT.jar

  /* ... new cell ... */

  import org.apache.spark.SparkContext
  import org.apache.spark.broadcast.Broadcast
  import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegression}
  import org.apache.spark.ml.param.ParamMap
  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  import org.apache.spark.mllib.linalg.{Vector, Vectors}
  import org.apache.spark.rdd.RDD
  import org.apache.spark.sql.{DataFrame, Row}
  import wordpressworkshop._
  import scalaz.Scalaz._
  import Pimps._

  /* ... new cell ... */

  @transient val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  
  import sqlContext.implicits._
  
  val dataPath = "file:///Users/g09971710/src/wordpress-workshop/data/"

  /* ... new cell ... */

  val trainingRDD: RDD[(Double, Vector)] =
      sc.textFile(dataPath + "trainFeatures-final").map(_.split("\t").map(_.toDouble).toList match {
        case label :: vector => label -> Vectors.dense(vector.toArray)
      })

  /* ... new cell ... */

  val testRDD: RDD[((Long, Long), Vector)] =
        sc.textFile(dataPath + "testFeatures-final").map(_.split("\t").toList match {
          case userId :: postId :: vector =>
            (userId.toLong, postId.toLong) -> Vectors.dense(vector.map(_.toDouble).toArray)
        })

  /* ... new cell ... */

  val training: DataFrame =
    trainingRDD.filter(!_._2.toArray.exists(x => x.isInfinity || x.isNaN)).toDF("label", "features")

  /* ... new cell ... */

  val lrr = LogisticRegressionRecommender(training)

  /* ... new cell ... */

  lrr.model.weights

  /* ... new cell ... */

  val evaluationTest: RDD[(Long, Set[Long])] =
    sc.textFile(dataPath + "evaluation-final.csv").map(_.split(",").toList match {
      case List(userId, posts, _) => userId.toLong -> posts.split(" ").map(_.toLong).toSet
    })

  /* ... new cell ... */

  val testUserIdToPostLikesBV: Broadcast[Map[Long, Set[Long]]] = sc.broadcast(evaluationTest.collect().toMap)

  /* ... new cell ... */

  val testData: DataFrame = testRDD.map {
    case ((userId, postId), vector) => (userId, postId,
      if (testUserIdToPostLikesBV.value(userId)(postId)) 1.0 else 0.0, vector)
  }.toDF("userId", "postId", "label", "features")

  /* ... new cell ... */

  testData.show()

  /* ... new cell ... */

  @transient val metrics = lrr.metrics(testData)

  /* ... new cell ... */

  // Precision by threshold
  metrics.precisionByThreshold().collect().toMap

  /* ... new cell ... */

  // Recall by threshold
  metrics.recallByThreshold().collect().roundX(2).toMap

  /* ... new cell ... */

  // Precision-Recall Curve = recall = f(precision)
  metrics.pr().collect().interpolatePercentiles(25).roundX(4)

  /* ... new cell ... */

    // F-measure
      metrics.fMeasureByThreshold().collect().interpolatePercentiles(25).roundX(3)
      val beta = 0.5
      metrics.fMeasureByThreshold(beta).collect().interpolatePercentiles(25).roundX(3)

  /* ... new cell ... */

  // AUPRC
  metrics.areaUnderPR()
     

  /* ... new cell ... */

   // ROC Curve
  metrics.roc.collect().interpolateLinear(25).roundX(3)

  /* ... new cell ... */

  // AUROC
  val auROC = metrics.areaUnderROC

  /* ... new cell ... */

  val bestThreshold = thresholds.filter(!_.isNaN).zip(metrics.roc.collect()).filter(_._2 |> {
    case (fpr, tpr) => tpr > 0.7
  }).minBy(_._2._1)

  /* ... new cell ... */

  val bestThreshold = thresholds.zip(metrics.roc.collect()).filter(_._2 |> {
    case (fpr, tpr) => fpr < 0.1
  }).maxBy(_._2._2)._1

  /* ... new cell ... */

  val lrrMAPScore =
        MAP.tuneThreshold(50, 100, 2)(lrr.likeScores(testData), evaluationTest).maxBy(_._2)

  /* ... new cell ... */

  val tagMAPScore = MAP.tuneThreshold(1, 100, 2)(TagLikelihoodRecommender.likeScores(testData), evaluationTest)

  /* ... new cell ... */

  tagMAPScore.maxBy(_._2)

  /* ... new cell ... */

  tagMAPScore

  /* ... new cell ... */

  tagMAPScore.

  /* ... new cell ... */
}
              