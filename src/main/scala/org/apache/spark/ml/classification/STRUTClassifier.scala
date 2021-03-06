package org.apache.spark.ml.classification
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.{RichDecisionTreeClassificationModel, SERTransfer, STRUTTransfer, TransferRandomForest}
import org.apache.spark.ml.util.{Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class STRUTClassifier(source: RichRandomForestClassificationModel) extends SingleSourceModelTransfer {

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

    val transferTrees = STRUTTransfer
      .transferModels(
        source._trees.map(_.asInstanceOf[RichDecisionTreeClassificationModel]),
        oldDataset,
        strategy,
        getNumTrees,
        "all",
        getSeed,
        Some(instr)
      )

    val m = new RandomForestClassificationModel(
      uid,
      transferTrees.map(_.asInstanceOf[DecisionTreeClassificationModel]),
      source.numFeatures,
      source.numClasses
    )
    instr.logSuccess(m)
    m
  }
}
