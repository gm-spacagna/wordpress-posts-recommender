package wordpressworkshop

case class StatsBlog(blogId: Long, numLikes: Long, numPosts: Long)

case class StatsUser(userId: Long, numLikes:Long, likeBlogs: Map[Long, Long])

case class BlogPost(blogId: Long, postId: Long,
                    authorId: Long, blogName: String, title: Option[String], content: Option[String],
                    language: String, categories: Set[String], tags: Set[String])
