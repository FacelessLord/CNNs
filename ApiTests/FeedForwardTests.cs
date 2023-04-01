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
        var nn = new FeedForward(new double[][,]
        {
            new double[,]
            {
                { 1, 1 },
                { -1, +1 }
            }
        });

        var inputs = new double[200][];
        var outputs = new double[200][];
        for (var i = 0; i < inputs.Length; i++)
        {
            inputs[i] = new[] { i / 200d, (200 - i) / 200d };
            outputs[i] = new[] { (200 - i) / 200d, i / 200d };
        }
        var error = 1d;
        for (var i = 0; i < 10000 && error > 0.05d; i++)
        {
            error = nn.LearnOn(inputs, outputs, 0.01);
            if (i % 100 == 0)
                Console.WriteLine($"Step {i}. Error is {error}");
        }
        Console.WriteLine(string.Join("\n", nn.Weights.Stringify()[0]));
    }
}