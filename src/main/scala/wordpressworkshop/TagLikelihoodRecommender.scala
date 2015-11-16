package wordpressworkshop

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

case object TagLikelihoodRecommender {
  def likeScores(testData: DataFrame): RDD[(Long, Long, Double)] =
    testData.map(row => (row.getAs[Long]("userId"), row.getAs[Long]("postId"), row.getAs[Vector]("features")(1)))

  def metrics(testData: DataFrame) = {
    val predictionAndLabels: RDD[(Double, Double)] =
      testData.map(row => row.getAs[Vector]("features")(1) -> row.getAs[Double]("label"))

    new BinaryClassificationMetrics(predictionAndLabels)
  }
}
