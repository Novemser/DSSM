package strlet.experiments;

import weka.core.Instance;

public class LowResExperiment extends AbstractMnistExperiment {

	private static final int PIXEL_SIZE = 4;

	protected Instance manipulateImage(Instance instance) {
		double[] intensities = instance.toDoubleArray();
		double[] newIntensities = new double[intensities.length];

		for (int r = 0; r < 28; r += PIXEL_SIZE) {
			for (int c = 0; c < 28; c += PIXEL_SIZE) {

				double sum = 0;
				for (int i = 0; i < PIXEL_SIZE; ++i) {
					for (int j = 0; j < PIXEL_SIZE; ++j) {
						double value = intensities[(r + i) * 28 + (c + j)];
						sum += value;
					}
				}

				double value = sum / (PIXEL_SIZE * PIXEL_SIZE);
				for (int i = 0; i < PIXEL_SIZE; ++i) {
					for (int j = 0; j < PIXEL_SIZE; ++j) {
						newIntensities[(r + i) * 28 + (c + j)] = Math
								.round(value);
					}
				}
			}
		}

		// Remember the class value
		newIntensities[intensities.length - 1] = intensities[intensities.length - 1];
		Instance i = new Instance(instance.weight(), newIntensities);
		i.setDataset(instance.dataset());
		return i;
	}

	public static void main(String[] args) throws Exception {
		LowResExperiment experiment = new LowResExperiment();
		experiment.runExperiment();
	}

}
