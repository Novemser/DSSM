package org.apache.spark.ml.classification

import java.util.concurrent.ThreadLocalRandom

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.{RichDecisionTreeClassificationModel, SERTransfer, STRUTTransfer}
import org.apache.spark.ml.util.{Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class MixClassifier(source: RichRandomForestClassificationModel) extends SingleSourceModelTransfer {
  setNumTrees(source._trees.length)
  override def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    logInfo(s"Transferring $getNumTrees trees")
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require(
        $(thresholds).length == numClasses,
        this.getClass.getSimpleName +
          ".train() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}"
      )
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, Algo.Classification, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(
      labelCol,
      featuresCol,
      predictionCol,
      probabilityCol,
      rawPredictionCol,
      impurity,
      numTrees,
      featureSubsetStrategy,
      maxDepth,
      maxBins,
      maxMemoryInMB,
      minInfoGain,
      minInstancesPerNode,
      seed,
      subsamplingRate,
      thresholds,
      cacheNodeIds,
      checkpointInterval
    )

    require(getNumTrees > 1, "There must be at least 2 trees for transfer learning in MIX classifier.")

    // Partition randomly into two group
    val serTrees = source._trees.take((getNumTrees + 1) / 2)
    val strutTrees = source._trees.takeRight(getNumTrees / 2)

    val transferredStrutTrees = STRUTTransfer
      .transferModels(
        strutTrees.map(_.asInstanceOf[RichDecisionTreeClassificationModel]),
        oldDataset,
        strategy,
        strutTrees.length,
        "all",
        getSeed,
        Some(instr)
      )

    val transferredSearTrees = SERTransfer
      .transferModels(
        serTrees.map(_.asInstanceOf[RichDecisionTreeClassificationModel]),
        oldDataset,
        strategy,
        serTrees.length,
        getFeatureSubsetStrategy,
        getSeed,
        Some(instr)
      )

    val transferTrees = transferredStrutTrees ++ transferredSearTrees

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestClassificationModel(
      uid,
      transferTrees.map(_.asInstanceOf[DecisionTreeClassificationModel]),
      numFeatures,
      math.min(source.numClasses, numClasses)
    )
    instr.logSuccess(m)
    m
  }
}
