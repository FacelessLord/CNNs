using Cloo;
using SilverHorn.Cloo.Command;
using SilverHorn.Cloo.Context;
using SilverHorn.Cloo.Device;
using SilverHorn.Cloo.Factories;
using SilverHorn.Cloo.Platform;

namespace Apis;

public class OpenClTestig
{
    public long Count(int dim)
    {
        var a = new long[1];
        var builder = new OpenCL200Factory();
        var Device = ComputePlatform.Platforms[0].Devices[0];
        var Properties = new List<ComputeContextProperty>
        {
            new ComputeContextProperty(ComputeContextPropertyName.Platform, Device.Platform.Handle.Value)
        };
        using (var Context = builder.CreateContext(ComputeDeviceTypes.All, Properties, null, IntPtr.Zero))
        {
            using (var Program = builder.BuildComputeProgram(Context, Text))
            {
                var Devs = new List<IComputeDevice>() { Device };
                try
                {
                    Program.Build(Devs, "", null, IntPtr.Zero);
                } catch (Exception e)
                {
                    Console.Error.WriteLine(e.StackTrace);
                    var log = Program.GetBuildLog(Device);
                    Console.WriteLine(log);
                    return 0;
                }
                var kernel = builder.CreateKernel(Program, "count");
                using (ComputeBuffer<long>
                       i = new ComputeBuffer<long>(Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, a))
                {
                    kernel.SetMemoryArgument(0, i);
                    using (var Queue = new ComputeCommandQueue(Context, Device, ComputeCommandQueueFlags.None))
                    {
                        Queue.Execute(kernel, null, new long[] { 1 }, null, null);
                        a = Queue.Read(i, true, 0, 1, null);
                    }
                }
            }
        }
        Console.WriteLine(a[0]);
        return a[0];
    }

    public static string Text = 
        @"__kernel void count(__global long* result)
{
    for(long i=0; i<1e+12; i++){
        result[0]++;
    }
}";
}