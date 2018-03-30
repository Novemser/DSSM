package org.apache.spark.ml.classification

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.DecisionTreeClassifierParams
import org.apache.spark.ml.tree.impl.{RandomForest, RichDecisionTreeClassificationModel, TransferRandomForest}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, Instrumentation, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class CustomDecisionTreeClassifier(val uid: String)
  extends ProbabilisticClassifier[Vector, CustomDecisionTreeClassifier, DecisionTreeClassificationModel]
  with DecisionTreeClassifierParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("decTree"))

  override def copy(extra: ParamMap): CustomDecisionTreeClassifier = {
    this
  }

  override protected def train(dataset: Dataset[_]): DecisionTreeClassificationModel = {
    println("Test train")
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)
    // Input: D, A, e
    // Output: DT model
    // 1. 如果D所有实例属于同一类别Ck, 则T为单节点树，并将类Ck作为该节点的标记，返回T
    // 2. 如果A=∅，则T为单节点树，并将D中实例数目最多的类Ck作为该节点的标记，返回T
    // 3. 否则，计算A中各个特征对于数据D的信息增益，选择增益最大的特征Ag
    // 4. 如果Ag的信息增益小于阈值e，T为单节点树，并将D中属于实例数最大的类Ck作为T的标记，返回T
    // 5. 否则对Ag中的所有可能取值a1, a2, a3...ai，根据这些将D分成集合D1, D2, D3...Di，选择
    //    Di中实例数量最多的类Ck作为标记分别生成子节点，由节点和子节点构成的树T，返回T
    // 6. 对第i个子节点，以Di为数据集，A-{Ag}为属性集，重复1-5，得到子树Ti，返回Ti
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy = getOldStrategy(categoricalFeatures, numClasses)
    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)

//    val Array(src, tgt) = oldDataset.randomSplit(Array(0.5, 0.5))
    val src = oldDataset
    val tgt = oldDataset

    val trees = TransferRandomForest.run(src, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = $(seed), instr = Some(instr), parentUID = Some(uid))

//    println(categoricalFeatures)
//    println(numClasses)
    val m = trees.head.asInstanceOf[DecisionTreeClassificationModel]
    instr.logSuccess(m)
    println("Transfer----------------------------------------------")
    TransferRandomForest.transfer(m.asInstanceOf[RichDecisionTreeClassificationModel], tgt, strategy, numTrees = 1,
      featureSubsetStrategy = "all", seed = $(seed), instr = Some(instr), parentUID = Some(uid))
    m
  }

  private[ml] def getOldStrategy(categoricalFeatures: Map[Int, Int],
                                 numClasses: Int): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity,
      subsamplingRate = 1.0)
  }
}

//class DTCModel private[ml](
//                          override val uid: String
//
//                          )
//extends ProbabilisticClassificationModel[Vector, DTCModel]
//with DecisionTreeModel with DecisionTreeClassifierParams with MLWritable with Serializable {
//
//}