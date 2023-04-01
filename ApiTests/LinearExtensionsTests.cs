using Apis.Math;
using Apis.Networks;
using FluentAssertions;

namespace ApiTests;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void MultipliesVectorByMatrix()
    {
        var vec = new double[] { 1, 2 };
        var matrix = new double[,]
        {
            { 1, 0, 1 },
            { 0, 1, 0 },
        };

        var result = vec.Times(matrix);

        result.Should().ContainInOrder(1, 2, 1);
    }

    [Test]
    public void DotProduct()
    {
        var vec = new double[] { 1, 2 };
        var operand = new double[] { 2, 1 };

        var result = vec.Dot(operand);

        result.Should().Be(4);
    }

    [Test]
    public void Scale()
    {
        var vec = new double[] { 1, 2 };

        var result = vec.Scale(4);

        result.Should().ContainInOrder(4, 8);
    }

    [Test]
    public void Sigmoid()
    {
        var vec = new double[] { 1, 2 };

        var result = vec.Sigmoid();

        result.Should().ContainInOrder(NNMath.Sigmoid(1), NNMath.Sigmoid(2));
    }
}