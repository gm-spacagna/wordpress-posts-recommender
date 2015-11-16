package wordpressworkshop

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.util.Random

object FalseLikes {

  def numLikeUsersDistributionArray(trainPostsRDD: RDD[(BlogPost, Set[Long])]): Array[Int] = {
    val bins = (0 to 100).toArray.map(_.toDouble)
    bins.zip(trainPostsRDD.map(_._2.size).histogram(bins)).flatMap {
      case (bin, count) => Array.fill(count.toInt)(bin.toInt)
    }
  }

  def blogPostsWithNonLikeUsers(trainPostsRDD: RDD[(BlogPost, Set[Long])],
                                numLikeUsersDistributionArrayBV: Broadcast[Array[Int]],
                                userIds: Broadcast[Set[Long]]): RDD[(BlogPost, Set[Long])] =
    trainPostsRDD.map {
      case (blogPost, users) =>
        val sum = numLikeUsersDistributionArrayBV.value.groupBy(identity).mapValues(_.length).values.sum
        val randomNumber: Int = Random.nextInt(sum.toInt)
        val nUsers = numLikeUsersDistributionArrayBV.value(randomNumber)
        val nonLikeUsers: Array[Long] = (userIds.value -- users).toArray
        blogPost -> Array.fill(nUsers)(nonLikeUsers(Random.nextInt(nonLikeUsers.length))).toSet
    }
}
