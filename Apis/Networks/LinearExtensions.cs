using Apis.Math;

namespace Apis.Networks;

public static class LinearExtensions
{
    public static double[] Times(this double[] source, double[,] matrix)
    {
        var resultSize = matrix.GetLength(1);
        var sourceSize = matrix.GetLength(0);
        var result = new double[resultSize];

        for (var i = 0; i < resultSize; i++)
        {
            for (var j = 0; j < sourceSize; j++)
            {
                result[i] += source[j] * matrix[j, i];
            }
        }

        return result;
    }
    public static double[][,] Scale(this double[][,] source, double scale)
    {
        var result = new double[source.Length][,];
        for (var i = 0; i < source.Length; i++)
        {
            result[i] = new double[source[i].GetLength(0), source[i].GetLength(1)];
            for (var j = 0; j < source[i].GetLength(0); j++)
            {
                for (var k = 0; k < source[i].GetLength(1); k++)
                {
                    result[i][j, k] = source[i][j, k] * scale;
                }
            }
        }
        return result;
    }

    public static double[][,] Add3Tensor(double[][,] a, double[][,] b)
    {
        var result = new double[a.Length][,];
        for (var i = 0; i < a.Length; i++)
        {
            result[i] = new double[a[i].GetLength(0), a[i].GetLength(1)];
            for (var j = 0; j < a[i].GetLength(0); j++)
            {
                for (var k = 0; k < a[i].GetLength(1); k++)
                {
                    result[i][j, k] = a[i][j, k] + b[i][j, k];
                }
            }
        }
        return result;
    }

    public static double Dot(this double[] source, double[] operand)
    {
        var resultSize = source.Length;
        var result = 0d;

        for (var i = 0; i < resultSize; i++)
        {
            result += source[i] * operand[i];
        }

        return result;
    }

    public static double[] Scale(this double[] source, double multiplier)
    {
        var resultSize = source.Length;
        var result = new double[resultSize];

        for (var i = 0; i < resultSize; i++)
        {
            result[i] += source[i] * multiplier;
        }

        return result;
    }

    public static double[] Sigmoid(this double[] source)
    {
        var resultSize = source.Length;
        var result = new double[resultSize];

        for (var i = 0; i < resultSize; i++)
        {
            result[i] = NNMath.Sigmoid(source[i]);
        }

        return result;
    }
}