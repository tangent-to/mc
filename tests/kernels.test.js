import { RBF, Matern32, Matern52, Periodic, Linear } from '../src/distributions/index.js';
import * as tf from '@tensorflow/tfjs-node';

describe('RBF Kernel', () => {
  test('constructor creates kernel with correct parameters', () => {
    const kernel = new RBF(1.0, 2.0);
    expect(kernel.lengthscale).toBe(1.0);
    expect(kernel.variance).toBe(2.0);
  });

  test('compute produces square matrix for single input', () => {
    const kernel = new RBF(1.0, 1.0);
    const X = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X);
    const shape = K.shape;

    expect(shape).toEqual([3, 3]);

    K.dispose();
    X.dispose();
  });

  test('diagonal elements are equal to variance', () => {
    const kernel = new RBF(1.0, 2.0);
    const X = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    expect(Karray[0][0]).toBeCloseTo(2.0, 5);
    expect(Karray[1][1]).toBeCloseTo(2.0, 5);
    expect(Karray[2][2]).toBeCloseTo(2.0, 5);

    K.dispose();
    X.dispose();
  });

  test('kernel value decreases with distance', () => {
    const kernel = new RBF(1.0, 1.0);
    const X = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    // k(0,0) > k(0,1) > k(0,2)
    expect(Karray[0][0]).toBeGreaterThan(Karray[0][1]);
    expect(Karray[0][1]).toBeGreaterThan(Karray[0][2]);

    K.dispose();
    X.dispose();
  });

  test('compute with two inputs', () => {
    const kernel = new RBF(1.0, 1.0);
    const X1 = tf.tensor2d([[0], [1]]);
    const X2 = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X1, X2);
    const shape = K.shape;

    expect(shape).toEqual([2, 3]);

    K.dispose();
    X1.dispose();
    X2.dispose();
  });

  test('kernel is symmetric', () => {
    const kernel = new RBF(1.0, 1.0);
    const X = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    expect(Karray[0][1]).toBeCloseTo(Karray[1][0], 5);
    expect(Karray[0][2]).toBeCloseTo(Karray[2][0], 5);
    expect(Karray[1][2]).toBeCloseTo(Karray[2][1], 5);

    K.dispose();
    X.dispose();
  });
});

describe('Matern32 Kernel', () => {
  test('constructor creates kernel with correct parameters', () => {
    const kernel = new Matern32(1.0, 2.0);
    expect(kernel.lengthscale).toBe(1.0);
    expect(kernel.variance).toBe(2.0);
  });

  test('diagonal elements are equal to variance', () => {
    const kernel = new Matern32(1.0, 2.0);
    const X = tf.tensor2d([[0], [1]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    expect(Karray[0][0]).toBeCloseTo(2.0, 5);
    expect(Karray[1][1]).toBeCloseTo(2.0, 5);

    K.dispose();
    X.dispose();
  });

  test('kernel value decreases with distance', () => {
    const kernel = new Matern32(1.0, 1.0);
    const X = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    expect(Karray[0][0]).toBeGreaterThan(Karray[0][1]);
    expect(Karray[0][1]).toBeGreaterThan(Karray[0][2]);

    K.dispose();
    X.dispose();
  });
});

describe('Matern52 Kernel', () => {
  test('constructor creates kernel with correct parameters', () => {
    const kernel = new Matern52(1.0, 2.0);
    expect(kernel.lengthscale).toBe(1.0);
    expect(kernel.variance).toBe(2.0);
  });

  test('Matern52 is smoother than Matern32', () => {
    const X = tf.tensor2d([[0], [1]]);

    const kernel32 = new Matern32(1.0, 1.0);
    const kernel52 = new Matern52(1.0, 1.0);

    const K32 = kernel32.compute(X);
    const K52 = kernel52.compute(X);

    const K32array = K32.arraySync();
    const K52array = K52.arraySync();

    // Matern52 should have higher off-diagonal values (smoother)
    expect(K52array[0][1]).toBeGreaterThan(K32array[0][1]);

    K32.dispose();
    K52.dispose();
    X.dispose();
  });
});

describe('Periodic Kernel', () => {
  test('constructor creates kernel with correct parameters', () => {
    const kernel = new Periodic(2.0, 1.0, 1.0);
    expect(kernel.period).toBe(2.0);
    expect(kernel.lengthscale).toBe(1.0);
    expect(kernel.variance).toBe(1.0);
  });

  test('kernel is periodic', () => {
    const kernel = new Periodic(1.0, 1.0, 1.0);

    // Points separated by period should have high correlation
    const X = tf.tensor2d([[0], [1], [2]]);
    const K = kernel.compute(X);
    const Karray = K.arraySync();

    // k(0,1) should be approximately equal to k(1,2) due to periodicity
    expect(Math.abs(Karray[0][1] - Karray[1][2])).toBeLessThan(0.1);

    K.dispose();
    X.dispose();
  });
});

describe('Linear Kernel', () => {
  test('constructor creates kernel with correct parameters', () => {
    const kernel = new Linear(2.0, 1.0);
    expect(kernel.variance).toBe(2.0);
    expect(kernel.offset).toBe(1.0);
  });

  test('kernel value increases with distance from offset', () => {
    const kernel = new Linear(1.0, 0.0);
    const X = tf.tensor2d([[0], [1], [2]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    // k(2,2) > k(1,1) > k(0,0) when offset=0
    expect(Karray[2][2]).toBeGreaterThan(Karray[1][1]);
    expect(Karray[1][1]).toBeGreaterThan(Karray[0][0]);

    K.dispose();
    X.dispose();
  });

  test('kernel at offset is zero', () => {
    const kernel = new Linear(1.0, 5.0);
    const X = tf.tensor2d([[5]]);

    const K = kernel.compute(X);
    const Karray = K.arraySync();

    expect(Karray[0][0]).toBeCloseTo(0, 5);

    K.dispose();
    X.dispose();
  });
});
