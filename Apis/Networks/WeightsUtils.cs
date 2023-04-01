using System.Globalization;
using System.Text;

namespace Apis.Networks;

internal static class WeightsUtils
{
    public static double[][,] CreatWeights(int[] layerConfig, Random rnd)
    {
        var layerCount = layerConfig.Length;
        var weights = new double[layerCount - 1][,];
        for (var i = 0; i < layerCount - 1; i++)
        {
            weights[i] = new double[layerConfig[i], layerConfig[i + 1]];
            for (var j = 0; j < layerConfig[i]; j++)
            {
                for (var k = 0; k < layerConfig[i + 1]; k++)
                {
                    weights[i][j, k] = rnd.NextDouble();
                }
            }
        }
        return weights;
    }

    public static string[][] Stringify(this double[][,] weights)
    {
        var result = new string[weights.Length][];
        for (var i = 0; i < weights.Length; i++)
        {
            result[i] = new string[weights[i].GetLength(0)];
            for (var j = 0; j < weights[i].GetLength(0); j++)
            {
                var builder = new StringBuilder(weights[i].GetLength(1) * 2);
                for (var k = 0; k < weights[i].GetLength(1); k++)
                {
                    builder.Append(weights[i][j, k].ToString(CultureInfo.InvariantCulture));
                    builder.Append(", ");
                }
                result[i][j] = builder.ToString();
            }
        }
        return result;
    }
}