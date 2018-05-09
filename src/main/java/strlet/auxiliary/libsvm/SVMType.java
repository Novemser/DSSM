package strlet.auxiliary.libsvm;

public enum SVMType {
	/** SVM type C-SVC (classification) */
	C_SVC,
	/** SVM type nu-SVC (classification) */
	NU_SVC,
	/** SVM type one-class SVM (classification) */
	ONE_CLASS_SVM,
	/** SVM type epsilon-SVR (regression) */
	EPSILON_SVR,
	/** SVM type nu-SVR (regression) */
	NU_SVR
}
