package wordpressworkshop

case object Pimps {
  implicit class PimpedArrayPoints2(points: Array[(Double, Double)]) {
    def interpolatePercentiles(n: Int): Array[(Double, Double)] = {
      val step = points.length.toDouble / (n - 1)

      points.zipWithIndex.groupBy{
        case (_, index) => (index.toDouble / step).round
      }
      .values.toList.map(_.map(_._1))
      .map(points => (points.map(_._1).sum / points.length, points.map(_._2).sum / points.length)).sortBy(_._1).toArray
    }

    def interpolateLinear(n: Int): Array[(Double, Double)] = {
      val xValues = points.map(_._1)

      val xMin = xValues.min
      val xMax = xValues.max
      val step = (xMax - xMin) / (n - 1)

      points.groupBy{
        case (x, _) => ((x - xMin) / step).round
      }
      .values.toList
      .map(points => (points.map(_._1).sum / points.length, points.map(_._2).sum / points.length)).sortBy(_._1).toArray
    }

    def roundX(nDigits: Int) = {
      val roundConst = math.pow(10, nDigits)

      points.groupBy {
        case (x, y) => math.round(x * roundConst) / roundConst
      }
      .mapValues(values => values.map(_._2).sum / values.length)
      .toList.sortBy(_._1).toArray
    }
  }
}