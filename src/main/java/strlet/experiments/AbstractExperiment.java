package strlet.experiments;

import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import strlet.auxiliary.ParallelBagging;
import strlet.auxiliary.libsvm.KernelType;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.TransferBagging;
import strlet.transferLearning.inductive.supervised.feda.Feda;
import strlet.transferLearning.inductive.taskLearning.consensusRegularization.ConsensusRegularization;
import strlet.transferLearning.inductive.taskLearning.naive.SourceOnly;
import strlet.transferLearning.inductive.taskLearning.naive.TargetOnly;
import strlet.transferLearning.inductive.taskLearning.svm.ASVM;
import strlet.transferLearning.inductive.taskLearning.trees.Mix;
import strlet.transferLearning.inductive.taskLearning.trees.Ser;
import strlet.transferLearning.inductive.taskLearning.trees.Strut;

public abstract class AbstractExperiment {

	// protected static int SEED = 9281743;
	public static int SEED = 1234;

	final protected double err(SingleSourceTransfer classifier, Instances test)
			throws Exception {
		int[][] confMtrx = calcConfMtrx(classifier, test);
		int total = 0;
		int correct = 0;
		for (int i = 0; i < confMtrx.length; ++i) {
			for (int j = 0; j < confMtrx.length; ++j) {
				total += confMtrx[i][j];
			}
		}
		for (int i = 0; i < confMtrx.length; ++i) {
			correct += confMtrx[i][i];
		}
		double accuracy = (correct + 0.0) / total;
		return 1 - accuracy;

	}

	final protected double berr(SingleSourceTransfer classifier, Instances test)
			throws Exception {
		int[][] confMtrx = calcConfMtrx(classifier, test);
		double[] localErr = new double[confMtrx.length];
		for (int i = 0; i < localErr.length; ++i) {
			int total = 0;
			int correct = confMtrx[i][i];
			for (int v : confMtrx[i]) {
				total += v;
			}
			localErr[i] = 1 - (0.0 + correct) / total;
		}
		return Utils.sum(localErr) / localErr.length;

	}

	private int[][] calcConfMtrx(SingleSourceTransfer classifier, Instances test)
			throws Exception {

		int[][] retVal = new int[test.numClasses()][test.numClasses()];
		for (int index = 0; index < test.numInstances(); ++index) {
			Instance instance = test.instance(index);
			int expected = (int) Math.round(instance.classValue());
			int observed = (int) Math.round(classifier
					.classifyInstance(instance));
			++retVal[expected][observed];
		}
		return retVal;
	}

	/**
	 * Change a number from probability to percentage (rounded to one decimal).
	 * 
	 * @param p
	 *            The probability to convert.
	 * @return The percentage calculated.
	 */
	protected double ToPerc(double p) {
		return Math.round(p * 1000) / 10.0;
	}

	public void runExperiment() throws Exception {

		ParallelBagging pb = new ParallelBagging();
		pb.setClassifier(new RandomTree());
		pb.setSeed(SEED);
		pb.setNumIterations(50);

		Feda feda = new Feda();
		feda.setBaseClassifier(pb);

		SourceOnly src = new SourceOnly();
		src.setBaseClassifier(pb);

		TargetOnly tgt = new TargetOnly();
		tgt.setBaseClassifier(pb);

		TransferBagging strut = new TransferBagging();
		strut.setClassifier(new Strut());
		strut.setSeed(SEED);
		strut.setNumIterations(50);

		TransferBagging ser = new TransferBagging();
		ser.setClassifier(new Ser());
		ser.setSeed(SEED);
		ser.setNumIterations(50);

		TransferBagging mix = new TransferBagging();
		mix.setClassifier(new Mix());
		mix.setSeed(SEED);
		mix.setNumIterations(50);

		ConsensusRegularization cs = new ConsensusRegularization();
		cs.setClassifier(new RandomTree());
		cs.setSeed(SEED);
		cs.setNumIterations(50);

		ASVM svm = new ASVM();
		// svm.setCost(Math.pow(2, c));
		// svm.setGamma(Math.pow(2, g));
		svm.setKernelType(new SelectedTag(KernelType.RBF.ordinal(),
				ASVM.TAGS_KERNELTYPE));

		// SingleSourceTransfer[] models = { src, tgt, feda };
		// SingleSourceTransfer[] models = { src, tgt };
		// SingleSourceTransfer[] models = { feda };
		// SingleSourceTransfer[] models = {cs};
//		SingleSourceTransfer[] models = { src, tgt, strut, ser, mix, cs, feda, svm };
		SingleSourceTransfer[] models = { src, tgt, ser, strut };

		runExperiment(models);

	}

	public abstract void runExperiment(SingleSourceTransfer[] models)
			throws Exception;

}
