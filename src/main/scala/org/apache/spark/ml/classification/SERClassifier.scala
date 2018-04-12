package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.{RichDecisionTreeClassificationModel, SERTransfer, TransferRandomForest}
import org.apache.spark.ml.util.{Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class SERClassifier(source: RichRandomForestClassificationModel)
  extends SingleSourceModelTransfer {

  setNumTrees(source._trees.length)

  protected override def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    logWarning(s"Transferring $getNumTrees trees")
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, Algo.Classification, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(labelCol, featuresCol, predictionCol, probabilityCol, rawPredictionCol,
      impurity, numTrees, featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, thresholds, cacheNodeIds, checkpointInterval)

    val transferTrees = SERTransfer
      .transferModels(source._trees.map(_.asInstanceOf[RichDecisionTreeClassificationModel]),
        oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestClassificationModel(uid, transferTrees.map(_.asInstanceOf[DecisionTreeClassificationModel]),
      numFeatures, numClasses)
    instr.logSuccess(m)
    m
  }
}
