package strlet.auxiliary.libsvm.kernel;

import strlet.auxiliary.libsvm.SVMParameter;
import strlet.auxiliary.libsvm.SVMProblem;

public class SVR_Q extends Kernel {

	private final int l;
	private final Cache cache;
	private final byte[] sign;
	private final int[] index;
	private int next_buffer;
	private float[][] buffer;
	private final double[] QD;

	public SVR_Q(SVMProblem prob, SVMParameter param) {
		super(prob.l, prob.x, param);
		l = prob.l;
		cache = new Cache(l, (long) (param.cache_size * (1 << 20)));
		QD = new double[2 * l];
		sign = new byte[2 * l];
		index = new int[2 * l];
		for (int k = 0; k < l; k++) {
			sign[k] = 1;
			sign[k + l] = -1;
			index[k] = k;
			index[k + l] = k;
			QD[k] = kernel_function(k, k);
			QD[k + l] = QD[k];
		}
		buffer = new float[2][2 * l];
		next_buffer = 0;
	}

	@Override
	public void swap_index(int i, int j) {
		do {
			byte _x = sign[i];
			sign[i] = sign[j];
			sign[j] = _x;
		} while (false);
		do {
			int _x = index[i];
			index[i] = index[j];
			index[j] = _x;
		} while (false);
		do {
			double _x = QD[i];
			QD[i] = QD[j];
			QD[j] = _x;
		} while (false);
	}

	@Override
	public float[] get_Q(int i, int len) {
		float[][] data = new float[1][];
		int j, real_i = index[i];
		if (cache.get_data(real_i, data, l) < l) {
			for (j = 0; j < l; j++)
				data[0][j] = (float) kernel_function(real_i, j);
		}

		// reorder and copy
		float buf[] = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		byte si = sign[i];
		for (j = 0; j < len; j++)
			buf[j] = (float) si * sign[j] * data[0][index[j]];
		return buf;
	}

	@Override
	public double[] get_QD() {
		return QD;
	}

}
