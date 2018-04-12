package org.apache.spark.ml.tree.impl

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.rdd.RDD

trait ModelTransfer extends Logging {
  def transferModels(trainedModels: Array[RichDecisionTreeClassificationModel],
                     target: RDD[LabeledPoint],
                     strategy: Strategy,
                     numTrees: Int,
                     featureSubsetStrategy: String,
                     seed: Long,
                     instr: Option[Instrumentation[_]],
                     parentUID: Option[String] = None): Array[RichDecisionTreeClassificationModel]
}
