package strlet.auxiliary;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class ThreadPool {

	private static int numThreads = 1;
	private static ExecutorService pool = Executors.newFixedThreadPool(numThreads);

	public static synchronized void initialize(int numThreads) {
		if (numThreads == ThreadPool.numThreads)
			return;

		pool.shutdown();
		try {
			boolean waiting = true;
			while (waiting) {
				waiting = !pool.awaitTermination(10, TimeUnit.MINUTES);
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		ThreadPool.numThreads = numThreads;
		pool = Executors.newFixedThreadPool(numThreads);
	}

	public static synchronized int poolSize() {
		return numThreads;
	}

	public static synchronized Future<?> submit(Runnable task) {
		return pool.submit(task);
	}

	public static synchronized <T> Future<T> submit(Runnable task, T result) {
		return pool.submit(task, result);
	}

	public static synchronized <T> Future<T> submit(Callable<T> task) {
		return pool.submit(task);
	}

	public static synchronized void shutdown() {
		numThreads = 0;
		initialize(1);
	}

	public static synchronized <T> List<Future<T>> invokeAll(
			Collection<? extends Callable<T>> tasks)
			throws InterruptedException {
		return pool.invokeAll(tasks);
	}

	public static synchronized void resetThreads() throws InterruptedException {

		final int before = numThreads;
		ThreadPool.shutdown();
		System.gc();
		Thread.sleep(1000);
		ThreadPool.initialize(before);

	}
}
