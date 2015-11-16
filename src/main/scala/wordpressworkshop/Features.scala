package wordpressworkshop

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import scalaz.Scalaz._

case class Features(categoriesLikelihood: Double, tagsLikelihood: Double, languageLikelihood: Double,
                    authorLikelihood: Double, titleLengthMeanError: Double,
                    blogLikelihood: Double,
                    averageLikesPerPost: Double)

case object Features {
  def blogIdToPriorBlogLikelihoodBV(statsUserRDD: RDD[StatsUser]): Map[Long, Double] =
    (statsUserRDD.map {
      case StatsUser(_, numLikes: Long, likeBlogs: Map[Long, Long]) => (likeBlogs, numLikes)
    }.reduce(_ |+| _) match {
      case (likeBlogs, numLikes) => likeBlogs.mapValues(_.toDouble / numLikes).map(identity)
    }).withDefaultValue(0.0)

  def meanBlogLikesPerPost(statsBlogRDD: RDD[StatsBlog]): Double = statsBlogRDD.map {
    case StatsBlog(_, numLikes: Long, numPosts: Long) => (numLikes, numPosts)
  }.reduce(_ |+| _) match {
    case (numLikes, numPosts) => numLikes.toDouble / numPosts
  }

  def userIdToOtherLikelihoodMaps(trainPostsRDD: RDD[(BlogPost, Set[Long])]): RDD[(Long, (Map[String, Int], Map[String, Int], Map[String, Int], Map[Long, Int], Map[Int, Int]))] =
    (for {
      (blogPost, users) <- trainPostsRDD
      userId <- users
    } yield userId ->(blogPost.categories.map(_ -> 1).toMap,
        blogPost.tags.map(_ -> 1).toMap,
        Map(blogPost.language -> 1),
        Map(blogPost.authorId -> 1),
        Map(blogPost.title.map(_.split("[^\\w']+").size).getOrElse(0) -> 1))
      )
    .reduceByKey(_ |+| _)
    .mapValues {
      case (categoriesLikelihoodMap, tagsLikelihoodMap,
      languageLikelihoodMap, authorLikelihoodMap,
      titleLengthLikelihoodMap) => (categoriesLikelihoodMap.toList.sortBy(_._2).takeRight(100).toMap,
        tagsLikelihoodMap.toList.sortBy(_._2).takeRight(100).toMap,
        languageLikelihoodMap,
        authorLikelihoodMap.toList.sortBy(_._2).takeRight(100).toMap,
        titleLengthLikelihoodMap)
    }

  def likelihoodSet(map: Map[String, Int], labels: Set[String]): Double =
    labels.flatMap(map.get).sum.toDouble / map.values.sum

  def likelihoodInt[K](map: Map[K, Int], label: K): Double =
    map.getOrElse(label, 0).toDouble / map.values.sum

  def likelihoodDouble[K](map: Map[K, Double], label: K): Double =
    map.getOrElse(label, 0.0) / map.values.sum

  def features(blogPostsAndUsers: RDD[(BlogPost, Set[Long])],
               userIdToOtherLikelihoodMaps: Broadcast[Map[Long, (Map[String, Int], Map[String, Int],
                 Map[String, Int], Map[Long, Int], Map[Int, Int])]],
               userIdToBlogLikelihood: Broadcast[Map[Long, Map[Long, Double]]],
               blogIdToPriorBlogLikelihoodBV: Broadcast[Map[Long, Double]],
               blogIdToAverageLikesPerPostBV: Broadcast[Map[Long, Double]],
               meanBlogLikesPerPost: Double) =
    for {
      (post, users) <- blogPostsAndUsers
      blogId = post.blogId
      postId = post.postId
      averageLikesPerPost = blogIdToAverageLikesPerPostBV.value.getOrElse(post.blogId, meanBlogLikesPerPost)
      userId <- users

      (categoriesLikelihoodMap,
      tagsLikelihoodMap,
      languageLikelihoodMap,
      authorLikelihoodMap,
      titleLengthLikelihoodMap) = userIdToOtherLikelihoodMaps.value(userId)

      titleLengthAverage = titleLengthLikelihoodMap.values.sum.toDouble / titleLengthLikelihoodMap.size

      blogLikelihoodMapOption = userIdToBlogLikelihood.value.get(userId)
      blogLikelihoodMap = blogLikelihoodMapOption.getOrElse(blogIdToPriorBlogLikelihoodBV.value)

    } yield (userId, post.postId) -> Features(
      categoriesLikelihood = likelihoodSet(categoriesLikelihoodMap, post.categories),
      tagsLikelihood = likelihoodSet(tagsLikelihoodMap, post.tags),
      languageLikelihood = likelihoodInt(languageLikelihoodMap, post.language),
      authorLikelihood = likelihoodInt(authorLikelihoodMap, post.authorId),
      titleLengthMeanError =
        math.abs(titleLengthAverage - post.title.map(_.split("[^\\w']+").size).getOrElse(0)),
      blogLikelihood = likelihoodDouble(blogLikelihoodMap, post.blogId),
      averageLikesPerPost = averageLikesPerPost
    )
}
