package strlet.transferLearning.inductive.supervised.trees;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.supervised.SingleSourceInstanceTransfer;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class MixedEntropy extends SingleSourceInstanceTransfer implements
		Randomizable, WeightedInstancesHandler {

	private int seed = 1;
	Node root = null;

	@Override
	public void setSeed(int seed) {
		this.seed = seed;
	}

	@Override
	public int getSeed() {
		return seed;
	}

	@Override
	public void buildModel(Instances source, Instances target) throws Exception {

		testWithFail(source, target);

		source = new Instances(source);
		source.deleteWithMissingClass();
		target = new Instances(target);
		target.deleteWithMissingClass();

		root = new Node(target);
		root.buildTree(source, target, new Random(seed));

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return root.distributionForInstance(instance);
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {
		MixedEntropy other = new MixedEntropy();
		other.setSeed(getSeed());
		if (root == null) {
			other.root = null;
		} else {
			other.root = root.makeDuplicate();
		}
		return other;
	}

	private class Node implements Serializable, Cloneable {

		private static final long serialVersionUID = -2635000774537783421L;

		/** Set of lambdas to use when testing TIG */
		protected final double[] LAMBDAS = { 0.75 };

		protected final Instances m_Header;

		/** Is the current node a leaf */
		protected boolean m_IsLeaf = false;

		/** Is there any useful information in the leaf */
		protected boolean m_IsEmpty = false;

		/** The subtrees appended to this tree. */
		protected Node[] m_Sons = null;

		/** The attribute to split on. */
		protected int m_Attribute = -1;

		/** The split point. */
		protected double m_SplitPoint = Double.NaN;

		/** Class probabilities from the training data. */
		protected double[] m_ClassDistribution = null;

		/** Training data distribution between sons. */
		protected double[] m_Weights = null;

		public Node(Instances data) {
			m_Header = new Instances(data, 0);
		}

		public Node makeDuplicate() {
			Node other = new Node(m_Header);
			other.m_IsLeaf = m_IsLeaf;
			other.m_IsEmpty = m_IsEmpty;
			other.m_Attribute = m_Attribute;
			other.m_SplitPoint = m_SplitPoint;
			if (m_Sons == null) {
				other.m_Sons = null;
			} else {
				other.m_Sons = new Node[m_Sons.length];
				for (int i = 0; i < m_Sons.length; ++i) {
					if (m_Sons[i] == null)
						other.m_Sons[i] = null;
					else
						other.m_Sons[i] = m_Sons[i].makeDuplicate();
				}
			}

			if (m_ClassDistribution == null) {
				other.m_ClassDistribution = null;
			} else {
				other.m_ClassDistribution = new double[m_ClassDistribution.length];
				for (int i = 0; i < m_ClassDistribution.length; ++i) {
					other.m_ClassDistribution[i] = m_ClassDistribution[i];
				}
			}
			
			if (m_Weights == null) {
				other.m_Weights = null;
			} else {
				other.m_Weights = new double[m_Weights.length];
				for (int i = 0; i < m_Weights.length; ++i) {
					other.m_Weights[i] = m_Weights[i];
				}
			}

			return other;
		}

		/**
		 * Computes class distribution of an instance using the decision tree.
		 * 
		 * @param instance
		 *            the instance to compute the distribution for
		 * @return the computed class distribution
		 * @throws Exception
		 *             if computation fails
		 */
		public double[] distributionForInstance(Instance instance) {

			if (m_IsLeaf && m_IsEmpty)
				return null;

			if (m_IsLeaf) {
				double[] normalizedDistribution = m_ClassDistribution.clone();
				Utils.normalize(normalizedDistribution);
				return normalizedDistribution;
			}

			if (instance.isMissing(m_Attribute)) { // Value is missing
				double[] dist = new double[m_Header.numClasses()];
				for (int i = 0; i < m_Sons.length; ++i) {
					double[] help = m_Sons[i].distributionForInstance(instance);
					if (help != null) {
						for (int j = 0; j < help.length; ++j) {
							dist[j] += m_Weights[i] * help[j];
						}
					}
				}
				return dist;
			}

			double[] d;
			if (m_Header.attribute(m_Attribute).isNominal()) {
				// For nominal attributes
				d = m_Sons[(int) Math.round(instance.value(m_Attribute))]
						.distributionForInstance(instance);
			} else {
				// For numeric attributes
				if (instance.value(m_Attribute) < m_SplitPoint) {
					d = m_Sons[0].distributionForInstance(instance);
				} else {
					d = m_Sons[1].distributionForInstance(instance);
				}
			}

			if (d != null) {
				return d;
			} else {
				double[] normalizedDistribution = m_ClassDistribution.clone();
				Utils.normalize(normalizedDistribution);
				return normalizedDistribution;
			}

		}

		public void buildTree(Instances source, Instances target, Random rand)
				throws Exception {

			// Check if empty
			if (target.numInstances() == 0) {
				m_IsLeaf = true;
				m_IsEmpty = true;
				return;
			}

			// Remember class distribution
			Distribution d = new Distribution(target);
			m_ClassDistribution = new double[target.numClasses()];
			for (int i = 0; i < m_ClassDistribution.length; ++i)
				m_ClassDistribution[i] = d.prob(i);

			// Check if pure or doesn't contain enough instances
			if (target.numInstances() == 1 || d.actualNumClasses() == 1) {
				m_IsLeaf = true;
				return;
			}

			boolean gainFaund = chooseAttribute(source, target, rand);
			if (!gainFaund) {
				m_IsLeaf = true;
				return;
			}

			// Split and calculate weights
			Instances[] sourceSubsets = split(source);
			Instances[] targetSubsets = split(target);
			m_Weights = new double[targetSubsets.length];
			for (int i = 0; i < m_Weights.length; ++i) {
				for (int index = 0; index < targetSubsets[i].numInstances(); ++index) {
					Instance instance = targetSubsets[i].instance(index);
					if (!instance.isMissing(m_Attribute)) {
						m_Weights[i] += instance.weight();
					}
				}
			}
			if (Utils.eq(0, Utils.sum(m_Weights))) {
				for (int i = 0; i < m_Weights.length; ++i) {
					m_Weights[i] = 1.0 / m_Weights.length;
				}
			} else {
				Utils.normalize(m_Weights);
			}

			// Build subtrees
			m_Sons = new Node[m_Weights.length];
			for (int i = 0; i < m_Sons.length; ++i) {
				m_Sons[i] = new Node(m_Header);
				m_Sons[i].buildTree(sourceSubsets[i], targetSubsets[i], rand);
			}

		}

		/**
		 * @param source
		 * @param target
		 * @param rand
		 * @return
		 */
		private boolean chooseAttribute(Instances source, Instances target,
				Random rand) {

			// Choose split index and value
			int[] window = new int[target.numAttributes()];
			for (int i = 0; i < window.length; ++i) {
				window[i] = i;
			}
			window[target.classIndex()] = window[window.length - 1];
			int windowSize = window.length - 1;

			double bestGain = Double.NEGATIVE_INFINITY;
			boolean gainFaund = false;
			int toCheck = (int) Math.round(Utils.log2(target.numAttributes()));
			while ((toCheck-- > 0) || !gainFaund) {

				// No more features to check
				if (windowSize == 0) {
					break;
				}

				int index = rand.nextInt(windowSize);
				int attr = window[index];
				window[index] = window[--windowSize];

				double splitPoint = Double.NaN;
				if (target.attribute(attr).isNumeric()) {
					splitPoint = findSplitPoint(source, target, attr);
					if (Double.isInfinite(splitPoint))
						continue;
				}

				double[][] sourceSplit = attrSplit(source, attr, splitPoint);
				double[][] targetSplit = attrSplit(target, attr, splitPoint);

				for (double lambda : LAMBDAS) {
					double gain = gain(sourceSplit, targetSplit, lambda);
					if (Utils.gr(gain, 0)) {
						gainFaund = true;
						if (Utils.gr(gain, bestGain)) {
							bestGain = gain;
							m_Attribute = attr;
							m_SplitPoint = splitPoint;
						}
					}
				}
			}

			return gainFaund;

		}

		private Instances[] split(Instances data) {

			if (data.attribute(m_Attribute).isNumeric()) {
				return numericSplit(data);
			} else {
				return nominalSplit(data);
			}
		}

		private Instances[] numericSplit(Instances data) {

			int[] counter = new int[2];
			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				if (instance.isMissing(m_Attribute)) {
					++counter[0];
					++counter[1];
				} else if (instance.value(m_Attribute) < m_SplitPoint) {
					++counter[0];
				} else {
					++counter[1];
				}
			}

			Instances[] retVal = new Instances[counter.length];
			for (int i = 0; i < retVal.length; ++i) {
				retVal[i] = new Instances(data, counter[i]);
			}

			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				if (instance.isMissing(m_Attribute)) {
					for (int i = 0; i < retVal.length; ++i)
						retVal[i].add(instance);
				} else if (instance.value(m_Attribute) < m_SplitPoint) {
					retVal[0].add(instance);
				} else {
					retVal[1].add(instance);
				}
			}

			return retVal;

		}

		private Instances[] nominalSplit(Instances data) {

			int[] counter = new int[data.attribute(m_Attribute).numValues()];
			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				if (instance.isMissing(m_Attribute)) {
					for (int i = 0; i < counter.length; ++i)
						++counter[i];
				} else {
					++counter[(int) Math.round(instance.value(m_Attribute))];
				}
			}

			Instances[] retVal = new Instances[counter.length];
			for (int i = 0; i < retVal.length; ++i) {
				retVal[i] = new Instances(data, counter[i]);
			}

			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				if (instance.isMissing(m_Attribute)) {
					for (int i = 0; i < retVal.length; ++i)
						retVal[i].add(instance);
				} else {
					retVal[(int) Math.round(instance.value(m_Attribute))]
							.add(instance);
				}
			}

			return retVal;
		}

		private double[][] attrSplit(Instances data, int attr, double splitPoint) {

			if (data.attribute(attr).isNumeric()) {
				return numericSplit(data, attr, splitPoint);
			} else {
				return nominalSplit(data, attr);
			}

		}

		private double[][] numericSplit(Instances data, int attr,
				double splitPoint) {

			double[][] dist = new double[2][data.numClasses()];
			double[] props = new double[2];

			// Split all instances with attr data
			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				double weight = instance.weight();
				int classValue = (int) Math.round(instance.classValue());
				if (instance.isMissing(attr)) {
					continue;
				} else if (instance.value(attr) < splitPoint) {
					dist[0][classValue] += weight;
				} else {
					dist[1][classValue] += weight;
				}
			}

			// Calculate weights of distribution
			for (int i = 0; i < props.length; ++i) {
				props[i] = Utils.sum(dist[i]);
			}
			if (Utils.eq(Utils.sum(props), 0)) {
				for (int i = 0; i < props.length; ++i) {
					props[i] = 1.0 / props.length;
				}
			} else {
				Utils.normalize(props);
			}

			// Add instances with missing data
			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				if (!instance.isMissing(attr)) {
					continue;
				}
				double weight = instance.weight();
				int classValue = (int) Math.round(instance.classValue());
				for (int i = 0; i < 2; ++i) {
					dist[i][classValue] += weight * props[i];
				}
			}

			return dist;

		}

		private double[][] nominalSplit(Instances data, int attr) {

			double[][] dist = new double[data.attribute(attr).numValues()][data
					.numClasses()];
			double[] props = new double[dist.length];

			// Split all instances with attr data
			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				double weight = instance.weight();
				int classValue = (int) Math.round(instance.classValue());
				if (instance.isMissing(attr)) {
					continue;
				} else {
					int attrValue = (int) Math.round(instance.value(attr));
					dist[attrValue][classValue] += weight;
				}
			}

			// Calculate weights of distribution
			for (int i = 0; i < props.length; ++i) {
				props[i] = Utils.sum(dist[i]);
			}
			if (Utils.eq(Utils.sum(props), 0)) {
				for (int i = 0; i < props.length; ++i) {
					props[i] = 1.0 / props.length;
				}
			} else {
				Utils.normalize(props);
			}

			// Add instances with missing data
			for (int index = 0; index < data.numInstances(); ++index) {
				Instance instance = data.instance(index);
				if (!instance.isMissing(attr)) {
					continue;
				}
				double weight = instance.weight();
				int classValue = (int) Math.round(instance.classValue());
				for (int i = 0; i < props.length; ++i) {
					dist[i][classValue] += weight * props[i];
				}
			}

			return dist;

		}

		private double findSplitPoint(Instances source, Instances target,
				int attr) {

			// Find attribute values
			Attribute attribute = m_Header.attribute(attr);
			if (!attribute.isNumeric()) {
				return Double.NaN;
			}
			source.sort(attribute);
			target.sort(attribute);
			double[] values = new double[source.numInstances()
					+ target.numInstances()];
			int counter = 0;
			for (int index = 0; index < source.numInstances(); ++index) {
				Instance instance = source.instance(index);
				if (instance.isMissing(attribute))
					break;
				values[counter++] = instance.value(attribute);
			}
			for (int index = 0; index < target.numInstances(); ++index) {
				Instance instance = target.instance(index);
				if (instance.isMissing(attribute))
					break;
				values[counter++] = instance.value(attribute);
			}
			for (int i = counter; i < values.length; ++i) {
				values[i] = Double.POSITIVE_INFINITY;
			}
			Arrays.sort(values);

			// Find first split
			double bestSplit = Double.NEGATIVE_INFINITY;
			int i;
			for (i = 1; i < counter - 1; ++i) {
				if (Double.isInfinite(values[i])) {
					return Double.NEGATIVE_INFINITY;
				}
				if (Utils.eq(values[i - 1], values[i])) {
					continue;
				}
				bestSplit = (values[i - 1] + values[i]) / 2;
				++i;
				break;
			}

			// Initialize source distribution
			double[][] sourceDist = new double[2][source.numClasses()];
			int sourceIndex = -1;
			for (int index = 0; index < source.numInstances(); ++index) {
				Instance instance = source.instance(index);
				int classValue = (int) Math.round(instance.classValue());
				if (instance.isMissing(attribute)) {
					sourceIndex = index;
					break;
				}
				if (instance.value(attribute) < bestSplit) {
					sourceDist[0][classValue] += instance.weight();
				} else {
					sourceDist[1][classValue] += instance.weight();
					if (sourceIndex == -1) {
						sourceIndex = index;
					}
				}
			}

			// Initialize target distribution
			double[][] targetDist = new double[2][target.numClasses()];
			int targetIndex = -1;
			for (int index = 0; index < target.numInstances(); ++index) {
				Instance instance = target.instance(index);
				int classValue = (int) Math.round(instance.classValue());
				if (instance.isMissing(attribute)) {
					targetIndex = index;
					break;
				}
				if (instance.value(attribute) < bestSplit) {
					targetDist[0][classValue] += instance.weight();
				} else {
					targetDist[1][classValue] += instance.weight();
					if (targetIndex == -1) {
						targetIndex = index;
					}
				}
			}

			// Calculate initial gain
			double bestGain = Double.NEGATIVE_INFINITY;
			for (double lambda : LAMBDAS) {
				double gain = gain(sourceDist, targetDist, lambda);
				if (Utils.gr(gain, bestGain)) {
					bestGain = gain;
				}
			}

			for (; i < counter - 1; ++i) {
				if (Double.isInfinite(values[i]))
					break;
				if (Utils.eq(values[i - 1], values[i]))
					continue;
				double splitPoint = (values[i - 1] + values[i]) / 2;

				while (sourceIndex != -1
						&& sourceIndex != source.numInstances()
						&& !source.instance(sourceIndex).isMissing(attribute)) {
					Instance instance = source.instance(sourceIndex);
					if (instance.value(attribute) < splitPoint) {
						int classValue = (int) Math
								.round(instance.classValue());
						sourceDist[0][classValue] += instance.weight();
						sourceDist[1][classValue] -= instance.weight();
						++sourceIndex;
					} else {
						break;
					}
				}

				while (targetIndex != -1
						&& targetIndex != target.numInstances()
						&& !target.instance(targetIndex).isMissing(attribute)) {
					Instance instance = target.instance(targetIndex);
					if (instance.value(attribute) < splitPoint) {
						int classValue = (int) Math
								.round(instance.classValue());
						targetDist[0][classValue] += instance.weight();
						targetDist[1][classValue] -= instance.weight();
						++targetIndex;
					} else {
						break;
					}
				}

				for (double lambda : LAMBDAS) {
					double gain = gain(sourceDist, targetDist, lambda);
					if (Utils.gr(gain, 0) && Utils.gr(gain, bestGain)) {
						bestGain = gain;
						bestSplit = splitPoint;
					}
				}
			}

			return bestSplit;

		}

		private double gain(double[][] source, double[][] target, double lambda) {

			double sourceIG = ContingencyTables.entropyOverColumns(source)
					- ContingencyTables.entropyConditionedOnRows(source);
			double targetIG = ContingencyTables.entropyOverColumns(target)
					- ContingencyTables.entropyConditionedOnRows(target);

			return (lambda * targetIG + (1 - lambda) * sourceIG);

		}

	}

}
