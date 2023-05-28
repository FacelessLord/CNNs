using Cloo.Bindings;
using SilverHorn.Cloo.Factories;

namespace Apis.Networks;

public class FeedForward : INeuralNetwork
{
    public int LayerCount { get; }
    private int[] LayerConfig { get; }
    public double[][,] Weights { get; private set; }

    public FeedForward(int[] layerConfig)
    {
        var rnd = new Random();
        LayerConfig = layerConfig;
        LayerCount = layerConfig.Length;
        Weights = WeightsUtils.CreatWeights(layerConfig, rnd);
    }

    public FeedForward(double[][,] weightsSetup)
    {
        LayerCount = weightsSetup.Length + 1;
        Weights = weightsSetup;
        LayerConfig = new int[] { weightsSetup[0].GetLength(0) }
            .Concat(weightsSetup.Select(w => w.GetLength(1)))
            .ToArray();
    }

    public double[] EvaluateSingle(double[] input)
    {
        var result = input;
        for (var i = 0; i < LayerCount - 1; i++)
        {
            result = result.Times(Weights[i]).Sigmoid();
        }
        return result;
    }

    public double[][] EvaluateBatch(double[][] inputs)
    {
        var results = new double[inputs.Length][];
        for (var i = 0; i < inputs.Length; i++)
        {
            var result = inputs[i];
            for (var j = 0; j < LayerCount - 1; j++)
            {
                result = result.Times(Weights[j]).Sigmoid();
            }
            results[i] = result;
        }
        return results;
    }

    public double LearnOn(double[][] inputs, double[][] outputs, double step)
    {
        var result = inputs.AsParallel().Select((input, sampleId) =>
        {
            var values = new double[LayerCount][];
            values[0] = input;
            for (var i = 1; i < LayerCount; i++)
            {
                values[i] = values[i - 1].Times(Weights[i - 1]).Sigmoid();
            }
            return (grad: GetGradient(values, outputs[sampleId]),
                Ln: ErrFunc(values[LayerCount - 1], outputs[sampleId]));
        }).Aggregate((p1, p2) => (LinearExtensions.Add3Tensor(p1.grad, p2.grad), p1.Ln + p2.Ln));

        Weights = LinearExtensions.Add3Tensor(Weights, result.grad.Scale(-step));

        return result.Ln;
    }

    private double ErrFunc(double[] real, double[] expected)
    {
        var result = 0d;
        for (int i = 0; i < real.Length; i++)
        {
            result += (real[i] - expected[i]) * (real[i] - expected[i]);
        }

        return result / 2;
    }


    private double[][,] GetGradient(double[][] values, double[] outputs)
    {
        var sigmaNets = values
            .Select(layer => layer.Select(v => v * (1 - v))
                .ToArray())
            .ToArray();
        var dLdW = new double[LayerCount][][][][];
        for (var a = 1; a < LayerCount; a++)
        {
            dLdW[a] = new double[LayerConfig[a]][][][];
            for (var m = 0; m < LayerConfig[a]; m++)
            {
                dLdW[a][m] = new double[LayerCount - 1][][];
                for (var i = 0; i < LayerCount - 1; i++)
                {
                    dLdW[a][m][i] = new double[LayerConfig[i]][];
                    for (var j = 0; j < LayerConfig[i]; j++)
                    {
                        dLdW[a][m][i][j] = new double[LayerConfig[i + 1]];
                        for (var k = 0; k < LayerConfig[i + 1]; k++)
                        {
                            if (1 + i > a || 1 + i == a && k != m)
                                dLdW[a][m][i][j][k] = 0;
                            else if (1 + i == a && k == m)
                                dLdW[a][m][i][j][k] = sigmaNets[a][m] * values[a - 1][j];
                            else
                            {
                                dLdW[a][m][i][j][k] = 0d;
                                for (var b = 0; b < LayerConfig[a - 1]; b++)
                                {
                                    dLdW[a][m][i][j][k] += sigmaNets[a][m] * Weights[a-1][b, m] * dLdW[a - 1][b][i][j][k];
                                }
                            }
                        }
                    }
                }
            }
        }

        var gradient = new double[LayerCount - 1][,];
        for (var i = 1; i < LayerCount; i++)
        {
            var layerGradient = new double[LayerConfig[i - 1], LayerConfig[i]];
            for (var j = 0; j < LayerConfig[i - 1]; j++)
            {
                for (var k = 0; k < LayerConfig[i]; k++)
                {
                    var s = 0d;
                    for (int m = 0; m < LayerConfig[LayerCount - 1]; m++)
                    {
                        s += (values[LayerCount - 1][m] - outputs[m]) * dLdW[LayerCount - 1][m][i-1][j][k];
                    }

                    layerGradient[j, k] = s;
                }
            }
            gradient[i - 1] = layerGradient;
        }
        return gradient;
    }
}