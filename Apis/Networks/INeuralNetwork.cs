namespace Apis.Networks;

public interface INeuralNetwork
{
    double[] EvaluateSingle(double[] input);

    double[][] EvaluateBatch(double[][] inputs);

    double LearnOn(double[][] inputs, double[][] outputs, double step);
}