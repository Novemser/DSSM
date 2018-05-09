package strlet.transferLearning.inductive.taskLearning.naive.trees;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.taskLearning.SingleSourceModelTransfer;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * <!-- globalinfo-start --> A simple tree transfer-learning algorithm. Training
 * is done by building a single unpruned C4.5 decision tree on the source
 * examples (using Wekas' J48 implementation), following a relabeling of the
 * leaves using the target examples.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * @author Noam Segev (nsegev@cs.technion.ac.il)
 *
 */
public class RelabelingTree extends SingleSourceModelTransfer implements
		Randomizable, WeightedInstancesHandler {

	private Node m_root = null;
	private int seed = 1;

	@Override
	public void setSeed(int seed) {
		this.seed = seed;
	}

	@Override
	public int getSeed() {
		return seed;
	}
	

	@Override
	protected void buildModel(Instances source) throws Exception {
		source = new Instances(source);
		source.deleteWithMissingClass();
		// Build source tree
		m_root = Node.buildTree(source, seed);
	}

	@Override
	protected void transferModel(Instances target) throws Exception {
		target = new Instances(target);
		target.deleteWithMissingClass();		
		m_root.relabel(target);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return m_root.distributionForInstance(instance);
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {
		RelabelingTree dup = new RelabelingTree();
		dup.setSeed(getSeed());
		if (m_root == null) {
			dup.m_root = null;
		} else {
			dup.m_root = m_root.makeDuplicate();
		}
		return dup;
	}

	private static class Node implements Serializable {

		private static final long serialVersionUID = -2752818564382895182L;

		private final Instances m_header;

		private final Random m_rand;

		/** Is the current node a leaf */
		protected boolean m_IsLeaf;

		/** Is there any useful information in the leaf */
		protected boolean m_IsEmpty;

		/** The subtrees appended to this tree. */
		protected Node[] m_Sons;

		/** The attribute to split on. */
		protected int m_Attribute;

		/** The split point. */
		protected double m_SplitPoint;

		/** Class probabilities from the training data. */
		protected double[] m_ClassDistribution;

		/** Training data distribution between sons. */
		protected double[] m_Weights;

		public Node(Instances data, Random rand) {
			m_header = new Instances(data, 0);
			m_rand = rand;
		}

		public static Node buildTree(Instances trainData, int seed)
				throws Exception {

			Node root = new Node(trainData, new Random(seed));
			root.buildTree(trainData);
			return root;
		}

		private void buildTree(Instances data) throws Exception {

			// Compute initial class counts
			double[] classProbs = new double[data.numClasses()];
			for (int i = 0; i < data.numInstances(); ++i) {
				Instance inst = data.instance(i);
				classProbs[(int) Math.round(inst.classValue())] += inst
						.weight();
			}

			// Create the attribute indices window
			int[] attIndicesWindow = new int[data.numAttributes() - 1];
			int j = 0;
			for (int i = 0; i < attIndicesWindow.length; ++i) {
				if (j == data.classIndex()) {
					++j; // do not include the class
				}
				attIndicesWindow[i] = j++;
			}

			buildTree(data, classProbs, attIndicesWindow);
		}

		private void buildTree(Instances data, double[] classProbs,
				int[] attIndicesWindow) throws Exception {

			m_IsLeaf = false;
			m_IsEmpty = false;
			m_Attribute = -1;
			m_ClassDistribution = null;
			m_Weights = null;
			m_SplitPoint = Double.NaN;
			m_Sons = null;

			// Make leaf if there are no training instances
			if (data.numInstances() == 0) {
				m_IsLeaf = true;
				m_IsEmpty = true;
				return;
			}

			// Check if node doesn't contain enough instances or is pure or
			// maximum depth reached
			m_ClassDistribution = (double[]) classProbs.clone();

			if ((Utils.sum(m_ClassDistribution) < 2) || isPure(classProbs)) {
				// Make leaf
				m_Attribute = -1;
				m_IsLeaf = true;
				m_Weights = null;
				return;
			}

			// Compute class distributions and value of splitting criterion for
			// each attribute
			double[] vals = new double[data.numAttributes()];
			double[][][] dists = new double[data.numAttributes()][0][0];
			double[][] props = new double[data.numAttributes()][0];
			double[] splits = new double[data.numAttributes()];

			// Investigate K random attributes
			int attIndex = 0;
			int windowSize = attIndicesWindow.length;
			int k = (int) Utils.log2(data.numAttributes()) + 1;
			boolean gainFound = false;
			while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

				int chosenIndex = m_rand.nextInt(windowSize);
				attIndex = attIndicesWindow[chosenIndex];

				// shift chosen attIndex out of window
				attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
				attIndicesWindow[windowSize - 1] = attIndex;
				windowSize--;

				splits[attIndex] = distribution(props, dists, attIndex, data);
				vals[attIndex] = gain(dists[attIndex]);

				if (Utils.gr(vals[attIndex], 0))
					gainFound = true;
			}

			// Find best attribute
			m_Attribute = Utils.maxIndex(vals);
			double[][] distribution = dists[m_Attribute];

			// Any useful split found?
			if (Utils.gr(vals[m_Attribute], 0)) {
				// Build subtrees
				m_SplitPoint = splits[m_Attribute];
				m_Weights = props[m_Attribute];
				Instances[] subsets = splitData(data);
				m_Sons = new Node[distribution.length];
				for (int i = 0; i < distribution.length; ++i) {
					m_Sons[i] = new Node(m_header, m_rand);
					m_Sons[i].buildTree(subsets[i], distribution[i],
							attIndicesWindow);
				}
			} else {
				// Make leaf
				m_Attribute = -1;
				m_IsLeaf = true;
			}

		}

		private boolean isPure(double[] classProbs) {

			double maxProb = classProbs[Utils.maxIndex(classProbs)];
			double sumProbs = Utils.sum(classProbs);
			return Utils.eq(maxProb, sumProbs);

		}

		/**
		 * Computes class distribution for an attribute.
		 *
		 * @param props
		 * @param dists
		 * @param att
		 *            the attribute index
		 * @param data
		 *            the data to work with
		 * @throws Exception
		 *             if something goes wrong
		 */
		private double distribution(double[][] props, double[][][] dists,
				int att, Instances data) throws Exception {

			double splitPoint = Double.NaN;
			Attribute attribute = data.attribute(att);
			double[][] dist = null;
			int indexOfFirstMissingValue = -1;

			if (attribute.isNominal()) {

				// For nominal attributes
				dist = new double[attribute.numValues()][data.numClasses()];
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(att)) {

						// Skip missing values at this stage
						if (indexOfFirstMissingValue < 0) {
							indexOfFirstMissingValue = i;
						}
						continue;
					}
					dist[(int) inst.value(att)][(int) inst.classValue()] += inst
							.weight();
				}
			} else {

				// For numeric attributes
				double[][] currDist = new double[2][data.numClasses()];
				dist = new double[2][data.numClasses()];

				// Sort data
				data.sort(att);

				// Move all instances into second subset
				for (int j = 0; j < data.numInstances(); j++) {
					Instance inst = data.instance(j);
					if (inst.isMissing(att)) {

						// Can stop as soon as we hit a missing value
						indexOfFirstMissingValue = j;
						break;
					}
					currDist[1][(int) inst.classValue()] += inst.weight();
				}

				// Value before splitting
				double priorVal = ContingencyTables
						.entropyOverColumns(currDist);

				// Save initial distribution
				for (int j = 0; j < currDist.length; j++) {
					System.arraycopy(currDist[j], 0, dist[j], 0, dist[j].length);
				}

				// Try all possible split points
				double currSplit = data.instance(0).value(att);
				double currVal, bestVal = -Double.MAX_VALUE;
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(att)) {

						// Can stop as soon as we hit a missing value
						break;
					}

					// Can we place a sensible split point here?
					if (inst.value(att) > currSplit) {

						// Compute gain for split point
						currVal = gain(currDist, priorVal);

						// Is the current split point the best point so far?
						if (currVal > bestVal) {

							// Store value of current point
							bestVal = currVal;

							// Save split point
							splitPoint = (inst.value(att) + currSplit) / 2.0;

							// Save distribution
							for (int j = 0; j < currDist.length; j++) {
								System.arraycopy(currDist[j], 0, dist[j], 0,
										dist[j].length);
							}
						}
					}
					currSplit = inst.value(att);

					// Shift over the weight
					currDist[0][(int) inst.classValue()] += inst.weight();
					currDist[1][(int) inst.classValue()] -= inst.weight();
				}
			}

			// Compute weights for subsets
			props[att] = new double[dist.length];
			for (int k = 0; k < props[att].length; k++) {
				props[att][k] = Utils.sum(dist[k]);
			}
			if (Utils.eq(Utils.sum(props[att]), 0)) {
				for (int k = 0; k < props[att].length; k++) {
					props[att][k] = 1.0 / (double) props[att].length;
				}
			} else {
				Utils.normalize(props[att]);
			}

			// Any instances with missing values ?
			if (indexOfFirstMissingValue > -1) {

				// Distribute weights for instances with missing values
				for (int i = indexOfFirstMissingValue; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (attribute.isNominal()) {

						// Need to check if attribute value is missing
						if (inst.isMissing(att)) {
							for (int j = 0; j < dist.length; j++) {
								dist[j][(int) inst.classValue()] += props[att][j]
										* inst.weight();
							}
						}
					} else {

						// Can be sure that value is missing, so no test
						// required
						for (int j = 0; j < dist.length; j++) {
							dist[j][(int) inst.classValue()] += props[att][j]
									* inst.weight();
						}
					}
				}
			}

			// Return distribution and split point
			dists[att] = dist;
			return splitPoint;
		}

		/**
		 * Computes value of splitting criterion after split.
		 *
		 * @param dist
		 *            the distributions
		 * @param priorVal
		 *            the splitting criterion
		 * @return the gain after the split
		 */
		private double gain(double[][] dist, double priorVal) {

			return priorVal - ContingencyTables.entropyConditionedOnRows(dist);
		}

		@SuppressWarnings("unchecked")
		public void relabel(Instances data) throws Exception {

			Queue<Node> splitNodes = new LinkedList<Node>();
			splitNodes.add(this);
			Queue<Instances> splitInstances = new LinkedList<Instances>();
			splitInstances.add(data);

			while (!splitNodes.isEmpty()) {
				Node treeNode = splitNodes.poll();
				data = splitInstances.poll();

				if (Utils.eq(data.sumOfWeights(), 0.0)) {
					treeNode.m_IsLeaf = true;
					treeNode.m_IsEmpty = true;
					treeNode.m_Attribute = -1;
					treeNode.m_ClassDistribution = null;
					treeNode.m_Weights = null;
					treeNode.m_SplitPoint = Double.NaN;
					treeNode.m_Sons = null;
					continue;
				} else {
					treeNode.m_IsEmpty = false;
				}

				treeNode.m_ClassDistribution = new double[data.numClasses()];
				for (int i = 0; i < treeNode.m_ClassDistribution.length; ++i) {
					treeNode.m_ClassDistribution[i] = 0;
				}
				Enumeration<Instance> enu = data.enumerateInstances();
				while (enu.hasMoreElements()) {
					Instance instance = enu.nextElement();
					double weight = instance.weight();
					int classValue = (int) Math.round(instance.classValue());
					treeNode.m_ClassDistribution[classValue] += weight;
				}

				// Change to leaf if all agree
				if (Utils.eq(Utils.sum(treeNode.m_ClassDistribution), m_ClassDistribution[Utils.maxIndex(m_ClassDistribution)])){
					treeNode.m_IsLeaf = true;
					treeNode.m_Weights = null;
					treeNode.m_Sons = null;
					continue;
				} else if (treeNode.m_IsLeaf) {
					treeNode.m_Weights = null;
					treeNode.m_Sons = null;
					continue;
				}


				Instances[] subsets = treeNode.splitData(data);
				data = null;
				for (int i = 0; i < subsets.length; ++i) {
					splitNodes.add(treeNode.m_Sons[i]);
					splitInstances.add(subsets[i]);
				}
			}
		}

		/**
		 * Splits instances into subsets based on the given split.
		 *
		 * @param data
		 *            the data to work with
		 * @return the subsets of instances
		 * @throws Exception
		 *             if something goes wrong
		 */
		private Instances[] splitData(Instances data) throws Exception {

			// Allocate array of Instances objects
			if (m_Weights.length < 0) {
				System.out.println();
			}
			Instances[] subsets = new Instances[m_Weights.length];
			for (int i = 0; i < m_Weights.length; i++) {
				subsets[i] = new Instances(data, data.numInstances());
			}

			// Go through the data
			for (int i = 0; i < data.numInstances(); i++) {

				// Get instance
				Instance inst = data.instance(i);

				// Does the instance have a missing value?
				if (inst.isMissing(m_Attribute)) {

					// Split instance up
					for (int k = 0; k < m_Weights.length; k++) {
						if (m_Weights[k] > 0) {
							Instance copy = (Instance) inst.copy();
							copy.setWeight(m_Weights[k] * inst.weight());
							subsets[k].add(copy);
						}
					}

					// Proceed to next instance
					continue;
				}

				// Do we have a nominal attribute?
				if (data.attribute(m_Attribute).isNominal()) {
					subsets[(int) inst.value(m_Attribute)].add(inst);

					// Proceed to next instance
					continue;
				}

				// Do we have a numeric attribute?
				if (data.attribute(m_Attribute).isNumeric()) {
					subsets[(inst.value(m_Attribute) < m_SplitPoint) ? 0 : 1]
							.add(inst);

					// Proceed to next instance
					continue;
				}

				// Else throw an exception
				throw new IllegalArgumentException("Unknown attribute type");
			}

			// Save memory
			for (int i = 0; i < m_Weights.length; i++) {
				subsets[i].compactify();
			}

			// Return the subsets
			return subsets;
		}

		private double gain(double[][] data) {
			return gain(data, ContingencyTables.entropyOverColumns(data));
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

			if (m_IsLeaf) {
				if (m_IsEmpty) {
					return null;
				}
				double[] normalizedDistribution = (double[]) m_ClassDistribution
						.clone();
				Utils.normalize(normalizedDistribution);
				return normalizedDistribution;
			}

			// double[] returnedDist = null;
			if (instance.isMissing(m_Attribute)) {
				// Value is missing
				double[] returnedDist = new double[m_header.numClasses()];
				// Split instance up
				for (int i = 0; i < m_Sons.length; i++) {
					double[] help = m_Sons[i].distributionForInstance(instance);
					if (help != null) {
						for (int j = 0; j < help.length; j++) {
							returnedDist[j] += m_Weights[i] * help[j];
						}
					}
				}
				return returnedDist;
			} else if (m_header.attribute(m_Attribute).isNominal()) {
				// For nominal attributes
				double[] returnedDist = m_Sons[(int) instance
						.value(m_Attribute)].distributionForInstance(instance);
				if (returnedDist != null) {
					return returnedDist;
				}
			} else {
				// For numeric attributes
				double[] returnedDist = null;
				if (instance.value(m_Attribute) < m_SplitPoint) {
					returnedDist = m_Sons[0].distributionForInstance(instance);
				} else {
					returnedDist = m_Sons[1].distributionForInstance(instance);
				}
				if (returnedDist != null) {
					return returnedDist;
				}
			}

			// Is node empty?
			if (m_ClassDistribution == null) {
				return null;
			}

			// Else return normalized distribution
			double[] normalizedDistribution = (double[]) m_ClassDistribution
					.clone();
			Utils.normalize(normalizedDistribution);
			return normalizedDistribution;

		}

		public Node makeDuplicate() {
			// TODO check and see if this is a correct use
			Node other = new Node(m_header, null);
			other.m_IsLeaf = this.m_IsLeaf;
			other.m_IsEmpty = this.m_IsEmpty;
			other.m_SplitPoint = this.m_SplitPoint;
			other.m_Attribute = this.m_Attribute;

			other.m_ClassDistribution = null;
			if (this.m_ClassDistribution != null) {
				other.m_ClassDistribution = Arrays.copyOf(
						this.m_ClassDistribution,
						this.m_ClassDistribution.length);
			}

			other.m_Weights = null;
			if (this.m_Weights != null) {
				other.m_Weights = Arrays.copyOf(this.m_Weights,
						this.m_Weights.length);
			}
			other.m_Sons = null;
			if (this.m_Sons != null) {
				other.m_Sons = new Node[this.m_Sons.length];
				for (int i = 0; i < this.m_Sons.length; ++i) {
					if (this.m_Sons[i] == null) {
						other.m_Sons[i] = null;
					} else {
						other.m_Sons[i] = this.m_Sons[i].makeDuplicate();
					}
				}
			}

			return other;
		}

		/**
		 * Prints tree structure.
		 *
		 * @return the tree structure
		 */
		@Override
		public String toString() {

			try {
				StringBuffer text = new StringBuffer();

				if (m_IsLeaf) {
					text.append(": ");
					text.append(m_header.classAttribute().value(
							Utils.maxIndex(m_ClassDistribution)));
				} else
					dumpTree(0, text);
				text.append("\n\nNumber of Leaves  : \t" + numLeaves() + "\n");
				text.append("\nSize of the tree : \t" + numNodes() + "\n");

				return text.toString();
			} catch (Exception e) {
				return "Can't print classification tree.";
			}
		}

		/**
		 * Help method for printing tree structure.
		 *
		 * @param depth
		 *            the current depth
		 * @param text
		 *            for outputting the structure
		 * @throws Exception
		 *             if something goes wrong
		 */
		private void dumpTree(int depth, StringBuffer text) throws Exception {

			for (int i = 0; i < m_Sons.length; ++i) {
				text.append("\n");
				for (int j = 0; j < depth; ++j)
					text.append("|   ");
				text.append(m_header.attribute(m_Attribute).name());
				if (m_header.attribute(i).isNominal())
					text.append(" = "
							+ m_header.attribute(m_Attribute).value(i));
				else if (i == 0)
					text.append(" <= " + Utils.doubleToString(m_SplitPoint, 6));
				else
					text.append(" > " + Utils.doubleToString(m_SplitPoint, 6));

				if (m_Sons[i].m_IsLeaf) {
					text.append(": ");
					if (m_Sons[i].m_IsEmpty) {
						text.append(m_header.classAttribute().value(
								Utils.maxIndex(m_ClassDistribution)));
					} else {
						text.append(m_header.classAttribute().value(
								Utils.maxIndex(m_Sons[i].m_ClassDistribution)));
					}
				} else
					m_Sons[i].dumpTree(depth + 1, text);
			}
		}

		/**
		 * Returns number of leaves in tree structure.
		 *
		 * @return the number of leaves
		 */
		public int numLeaves() {

			if (m_IsLeaf)
				return 1;

			int num = 0;
			for (int i = 0; i < m_Sons.length; ++i)
				num += m_Sons[i].numLeaves();

			return num;
		}

		/**
		 * Returns number of nodes in tree structure.
		 *
		 * @return the number of nodes
		 */
		public int numNodes() {

			int no = 1;

			if (!m_IsLeaf)
				for (int i = 0; i < m_Sons.length; ++i)
					no += m_Sons[i].numNodes();

			return no;
		}

	}

}
