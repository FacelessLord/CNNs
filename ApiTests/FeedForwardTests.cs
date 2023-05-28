using Apis.Math;
using Apis.Networks;
using FluentAssertions;
using NUnit.Framework;

namespace ApiTests;

[TestFixture]
public class FeedForwardTests
{
    [Test]
    public void EvaluatesTwoLayerNetwork_Single()
    {
        var nn = new FeedForward(new double[1][,]
        {
            new double[,] { { 1, -1 } }
        });

        var result = nn.EvaluateSingle(new double[] { 2 });

        result.Should().ContainInOrder(NNMath.Sigmoid(2), NNMath.Sigmoid(-2));
    }

    [Test]
    public void EvaluatesThreeLayerNetwork_Single()
    {
        var nn = new FeedForward(new double[][,]
        {
            new double[,] { { 1, -1 } },

            new double[,]
            {
                { 1, 1 },
                { -1, +1 }
            }
        });

        var result = nn.EvaluateSingle(new double[] { 2 });
        var a1 = NNMath.Sigmoid(2);
        var a2 = NNMath.Sigmoid(-2);
        var b1 = NNMath.Sigmoid(a1 - a2);
        var b2 = NNMath.Sigmoid(a1 + a2);
        result.Should().ContainInOrder(b1, b2);
    }
    [Test]
    public void EvaluatesBatchThreeLayerNetwork_Single()
    {
        var nn = new FeedForward(new double[][,]
        {
            new double[,] { { 1, -1 } },

            new double[,]
            {
                { 1, 1 },
                { -1, +1 }
            }
        });

        var result = nn.EvaluateBatch(new[] { new double[] { 2 }, new double[] { 1 } });
        var result2Single = nn.EvaluateSingle(new double[] { 2 });
        var result1Single = nn.EvaluateSingle(new double[] { 1 });

        result[0].Should().Equal(result2Single);
        result[1].Should().Equal(result1Single);
    }

    [Test]
    public void NetworkLearnsToRevertInputs()
    {
        var nn = new FeedForward(new[] { 6, 12, 1 });

        const int B = 1 << 6;
        var inputs = new double[B*B][];
        var outputs = new double[B*B][];
        for (var i = 0; i < B; i++)
        {
            for (var k = 0; k < B; k++)
            {
                inputs[i * B + k] = new double[12];
                for (var j = 0; j < 6; j++)
                {
                    inputs[i][j] = (i >> j) % 2;
                }
                for (var j = 0; j < 6; j++)
                {
                    inputs[i][j + 6] = (k >> j) % 2;
                }
                
                outputs[i * B + k] = new double[6];
                var sum = i + k;
                for (var j = 0; j < 6; j++)
                {
                    outputs[i * B + k][j] = (sum >> j) % 2;
                }
            }
        }
        const int maxIter = 100000;
        var error = 1d;
        for (var i = 0; i < maxIter && error > 0.05d; i++)
        {
            error = nn.LearnOn(inputs, outputs, 0.1);
            if (i % (maxIter / 100) == 0)
            {
                Console.WriteLine($"Step {i}. Error is {error}");
                Console.Out.Flush();
            }
        }
        Console.WriteLine(string.Join("\n", nn.Weights.Stringify()[0]));
        if (nn.Weights.Length > 1)
            Console.WriteLine(string.Join("\n", nn.Weights.Stringify()[1]));
        if (nn.Weights.Length > 2)
            Console.WriteLine(string.Join("\n", nn.Weights.Stringify()[2]));
    }
}