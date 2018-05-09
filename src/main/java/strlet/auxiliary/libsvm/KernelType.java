package strlet.auxiliary.libsvm;

public enum KernelType {
	/** kernel type linear: u'*v */
	LINEAR,
	/** kernel type polynomial: (gamma*u'*v + coef0)^degree */
	POLYNOMIAL,
	/** kernel type radial basis function: exp(-gamma*|u-v|^2) */
	RBF,
	/** kernel type sigmoid: tanh(gamma*u'*v + coef0) */
	SIGMOID,
	/** Special case - no need to calculate a kernel */
	PRECOMPUTED
}
