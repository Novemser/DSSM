package strlet;

import strlet.auxiliary.ParallelBagging;
import strlet.auxiliary.UnderBagging;
import strlet.auxiliary.ThreadPool;
import strlet.auxiliary.libsvm.KernelType;
import strlet.experiments.AbstractExperiment;
import strlet.experiments.DigitsExperiment;
import strlet.experiments.InversionExperiment;
import strlet.experiments.LandminesExperiment;
import strlet.experiments.LetterExperiment;
import strlet.experiments.LowResExperiment;
import strlet.experiments.MushroomExperiment;
import strlet.experiments.NewsgroupsExperiment;
import strlet.experiments.OfficeCaltechExperiment;
import strlet.experiments.SRAAexperiment;
import strlet.experiments.UspsExperiment;
import strlet.experiments.WineExperiment;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.TransferBagging;
import strlet.transferLearning.inductive.supervised.bagging.TrBagg;
import strlet.transferLearning.inductive.supervised.boosting.TradaBoost;
import strlet.transferLearning.inductive.supervised.feda.Feda;
import strlet.transferLearning.inductive.supervised.trees.MixedEntropy;
import strlet.transferLearning.inductive.taskLearning.consensusRegularization.ConsensusRegularization;
import strlet.transferLearning.inductive.taskLearning.svm.ASVM;
import strlet.transferLearning.inductive.taskLearning.trees.Mix;
import strlet.transferLearning.inductive.taskLearning.trees.Ser;
import strlet.transferLearning.inductive.taskLearning.trees.Strut;
import weka.classifiers.trees.RandomTree;
import weka.core.SelectedTag;

public class Strlet {

	public static void main(String[] args) throws Exception {
		if ((args.length < 2) || (args.length > 3)) {
			System.err.println("Order of parameters must be <test> <algorithm> [#threads]");
			System.exit(1);
		}

		AbstractExperiment experiment = getExperiment(args[0]);
		boolean isUnbalanced = isUnbalanced(args[0]);
		SingleSourceTransfer[] classifiers = new SingleSourceTransfer[1];
		classifiers[0] = toClassifier(args[1], isUnbalanced);
		if (args.length == 3) {
			int threads = getThreadsCount(args[2]);
			ThreadPool.initialize(threads);
		}
		try {
			experiment.runExperiment(classifiers);
		} finally {
			ThreadPool.shutdown();
		}

	}

	private static SingleSourceTransfer toClassifier(String name, boolean isUnbalanced) throws Exception {
		if (name.equalsIgnoreCase("Feda")) {
			if (isUnbalanced) {
				UnderBagging ub = new UnderBagging();
				ub.setClassifier(new RandomTree());
				ub.setNumIterations(50);
				ub.setSeed(AbstractExperiment.SEED);
				Feda feda = new Feda();
				feda.setBaseClassifier(ub);
				return feda;				
			} else {
				ParallelBagging pb = new ParallelBagging();
				pb.setClassifier(new RandomTree());
				pb.setSeed(AbstractExperiment.SEED);
				pb.setNumIterations(50);
				Feda feda = new Feda();
				feda.setBaseClassifier(pb);
				return feda;				
			}
		} else if (name.equalsIgnoreCase("TradaBoost")) {
			TradaBoost tradaBoost = new TradaBoost();
			tradaBoost.setNumIterations(50);
			tradaBoost.setClassifier(new RandomTree());
			tradaBoost.setSeed(AbstractExperiment.SEED);
			return tradaBoost;
		} else if (name.equalsIgnoreCase("TrBagg")) {
			TrBagg trBagg = new TrBagg();
			trBagg.setClassifier(new RandomTree());
			trBagg.setSeed(AbstractExperiment.SEED);
			return trBagg;
		} else if (name.equalsIgnoreCase("ASVM")) {
			ASVM svm = new ASVM();
			// svm.setCost(Math.pow(2, c));
			// svm.setGamma(Math.pow(2, g));
			svm.setKernelType(new SelectedTag(KernelType.RBF.ordinal(), ASVM.TAGS_KERNELTYPE));
			return svm;
		} else if (name.equalsIgnoreCase("ConsensusRegularization")) {
			ConsensusRegularization cr = new ConsensusRegularization();
			cr.setClassifier(new RandomTree());
			cr.setNumIterations(50);
			cr.setSeed(AbstractExperiment.SEED);
			return cr;
		} else if (name.equalsIgnoreCase("Strut")) {
			TransferBagging strut = new TransferBagging();
			strut.setClassifier(new Strut());
			strut.setSeed(AbstractExperiment.SEED);
			strut.setNumIterations(50);
			return strut;
		} else if (name.equalsIgnoreCase("Ser")) {
			TransferBagging ser = new TransferBagging();
			ser.setClassifier(new Ser());
			ser.setSeed(AbstractExperiment.SEED);
			ser.setNumIterations(50);
			return ser;
		} else if (name.equalsIgnoreCase("Mix")) {
			TransferBagging mix = new TransferBagging();
			mix.setClassifier(new Mix());
			mix.setSeed(AbstractExperiment.SEED);
			mix.setNumIterations(50);
			return mix;
		} else if (name.equalsIgnoreCase("MixedEntropy")) {
			TransferBagging mixedEntropy = new TransferBagging();
			mixedEntropy.setClassifier(new MixedEntropy());
			mixedEntropy.setSeed(AbstractExperiment.SEED);
			mixedEntropy.setNumIterations(50);
			return mixedEntropy;
		}
		throw new IllegalArgumentException("Unknown algorithm " + name);
	}

	private static AbstractExperiment getExperiment(String name) {
		if (name.equalsIgnoreCase("Mushroom")) {
			return new MushroomExperiment();
		} else if (name.equalsIgnoreCase("Letter")) {
			return new LetterExperiment();
		} else if (name.equalsIgnoreCase("Wine")) {
			return new WineExperiment();
		} else if (name.equalsIgnoreCase("Digits")) {
			return new DigitsExperiment();
		} else if (name.equalsIgnoreCase("inversion")) {
			return new InversionExperiment();
		} else if (name.equalsIgnoreCase("HighRes")) {
			return new LowResExperiment();
		} else if (name.equalsIgnoreCase("USPS")) {
			return new UspsExperiment();
		} else if (name.equalsIgnoreCase("Landmine")) {
			return new LandminesExperiment();
		} else if (name.equalsIgnoreCase("Office")) {
			return new OfficeCaltechExperiment();
		} else if (name.equalsIgnoreCase("20NG")) {
			return new NewsgroupsExperiment("rec", "talk");
		} else if (name.equalsIgnoreCase("SRAA")) {
			return new SRAAexperiment("auto", "aviation");
		}

		throw new IllegalArgumentException("Unknown test");
	}
	
	private static boolean isUnbalanced(String name) {
		if (name.equalsIgnoreCase("Landmine")) {
			return true;
		}
		return false;
	}

	private static int getThreadsCount(String count) {
		return Integer.valueOf(count);
	}

}
