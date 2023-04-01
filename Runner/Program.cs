// See https://aka.ms/new-console-template for more information

using System.Globalization;
using Apis.Networks;

var nn = new FeedForward(new double[1][,]
{
    new double[,]
    {
        { -2.4612785247113123, 2.4612785247113123 },
        { 2.4630307345829436, -2.4630307345829436 },
    }
});
var res = nn.EvaluateSingle(new[] { 0d, 1 });
Console.WriteLine(string.Join(", ", nn.EvaluateSingle(new []{ 0d, 1}).Select(d => d.ToString(CultureInfo.InvariantCulture))));

Console.WriteLine(nn.LearnOn(new double[1][] { new[] { 0d, 1 } }, new double[1][] { new[] { 1d, 0 } }, 0.001));