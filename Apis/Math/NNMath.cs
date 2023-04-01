namespace Apis.Math;

public static class NNMath
{
    public static double Sigmoid(double src)
    {
        return 1 / (System.Math.Exp(-src) + 1);
    }
}