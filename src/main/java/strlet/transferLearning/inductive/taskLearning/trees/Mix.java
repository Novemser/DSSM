package strlet.transferLearning.inductive.taskLearning.trees;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.taskLearning.SingleSourceModelTransfer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.WeightedInstancesHandler;

public class Mix extends SingleSourceModelTransfer implements
		WeightedInstancesHandler, Randomizable {

	protected Ser m_ser = new Ser();
	protected Strut m_strut = new Strut();

	private int m_seed = 1;

	@Override
	public void setSeed(int seed) {
		m_seed = seed;
	}

	@Override
	public int getSeed() {
		return m_seed;
	}

	@Override
	protected void buildModel(Instances source) throws Exception {
		m_ser.setSeed(getSeed());
		m_ser.buildModel(source);
		m_strut.setSeed(getSeed());
		m_strut.buildModel(source);
	}

	@Override
	protected void transferModel(Instances target) throws Exception {
		m_ser.transferModel(target);
		m_strut.transferModel(target);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (instance.classAttribute().isNumeric()) {
			double[] dist = { 0 };
			dist[0] = (m_ser.classifyInstance(instance) + m_strut
					.classifyInstance(instance)) / 2;
			return dist;
		} else {
			double[] dist = new double[instance.numClasses()];
			double[] dist1 = m_ser.distributionForInstance(instance);
			double[] dist2 = m_strut.distributionForInstance(instance);
			for (int i = 0; i < dist.length; ++i) {
				dist[i] = (dist1[i] + dist2[i]) / 2;
			}
			return dist;
		}
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {
		Mix dup = new Mix();
		dup.setSeed(getSeed());
		dup.m_ser = Ser.class.cast(m_ser.makeDuplicate());
		dup.m_strut = Strut.class.cast(m_strut.makeDuplicate());
		return dup;
	}

}
