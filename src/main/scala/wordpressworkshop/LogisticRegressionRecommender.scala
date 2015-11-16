package wordpressworkshop

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

case class LogisticRegressionRecommender(training: DataFrame) {

  val lr = new LogisticRegression()
  val paramMap = ParamMap(lr.maxIter -> 20)
                 .put(lr.regParam -> 0.01)
                 .put(lr.probabilityCol -> "probability")

  val model: LogisticRegressionModel = lr.fit(training, paramMap)

  def metrics(testData: DataFrame) = {
    val predictionAndLabels: RDD[(Double, Double)] =
      model.transform(testData).map(row => row.getAs[Vector]("probability")(1) -> row.getAs[Double]("label"))

    new BinaryClassificationMetrics(predictionAndLabels)
  }

  def likeScores(testData: DataFrame): RDD[(Long, Long, Double)] =
    model.transform(testData)
    .map(row => (row.getAs[Long]("userId"), row.getAs[Long]("postId"), row.getAs[Vector]("probability")(1)))
}
