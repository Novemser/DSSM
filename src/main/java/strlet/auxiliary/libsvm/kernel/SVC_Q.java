package strlet.auxiliary.libsvm.kernel;

import strlet.auxiliary.libsvm.SVMParameter;
import strlet.auxiliary.libsvm.SVMProblem;

/**
 * Q matrices for various formulations
 */
public class SVC_Q extends Kernel {

	private final byte[] y;
	private final Cache cache;
	private final double[] QD;

	public SVC_Q(SVMProblem prob, SVMParameter param, byte[] y_) {
		super(prob.l, prob.x, param);
		y = (byte[]) y_.clone();
		cache = new Cache(prob.l, (long) (param.cache_size * (1 << 20)));
		QD = new double[prob.l];
		for (int i = 0; i < prob.l; i++)
			QD[i] = kernel_function(i, i);
	}

	@Override
	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int start, j;
		if ((start = cache.get_data(i, data, len)) < len) {
			for (j = start; j < len; j++)
				data[0][j] = (float) (y[i] * y[j] * kernel_function(i, j));
		}
		return data[0];
	}

	@Override
	public double[] get_QD() {
		return QD;
	}

	@Override
	public void swap_index(int i, int j) {
		cache.swap_index(i, j);
		super.swap_index(i, j);
		do {
			byte _x = y[i];
			y[i] = y[j];
			y[j] = _x;
		} while (false);
		do {
			double _x = QD[i];
			QD[i] = QD[j];
			QD[j] = _x;
		} while (false);
	}
}
