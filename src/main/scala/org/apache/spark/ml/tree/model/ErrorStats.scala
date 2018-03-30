package org.apache.spark.ml.tree.model

abstract class ErrorStats extends Serializable {
  def totalSampleCount(): Double

  def errorSampleCount(): Double

  def errorSampleValue(): Double

  def errorRate(): Double

  override def toString: String = {
    s"Total sample [${totalSampleCount()}], error sample [${errorSampleCount()}], error rate [${errorRate()}]"
  }
}

class ClassificationError(val rightCount: Double, val totalCount: Double) extends ErrorStats {
  override def totalSampleCount(): Double = totalCount

  override def errorSampleCount(): Double = totalCount - rightCount

  override def errorSampleValue(): Double = totalCount - rightCount

  override def errorRate(): Double = errorSampleCount() / totalSampleCount()
}

object ClassificationError {
  def apply(rightCount: Double, totalCount: Double): ClassificationError =
    new ClassificationError(rightCount, totalCount)
}