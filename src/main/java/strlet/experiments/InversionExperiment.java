package strlet.experiments;

import weka.core.Instance;

public class InversionExperiment extends AbstractMnistExperiment {

	protected Instance manipulateImage(Instance instance) {

		double[] intensities = instance.toDoubleArray();
		double[] newIntensities = new double[intensities.length];
		for (int i = 0; i < 28 * 28; ++i)
			newIntensities[i] = Math.round(255 - intensities[i]);

		// Remember the class value
		newIntensities[intensities.length - 1] = intensities[intensities.length - 1];
		Instance i = new Instance(instance.weight(), newIntensities);
		i.setDataset(instance.dataset());
		return i;
	}

	public static void main(String[] args) throws Exception {
		InversionExperiment experiment = new InversionExperiment();
		experiment.runExperiment();
	}

}
