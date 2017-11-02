using System;
using System.Collections.Generic;
using System.IO;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;

using Encog.ML.Data;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.ML.Train;
using Encog.ML.Data.Basic;
using Encog;


namespace Income
{

    internal class Program
    {
        public static double[][] networkInput;

        private static void ReadCSV(ref List<double> listA, ref List<double> listB)
        {
            using (var reader = new StreamReader(@".\incomeedu.csv"))
            {
                listA = new List<double>();
                listB = new List<double>();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    listA.Add(Convert.ToDouble(values[0]));
                    listB.Add(Convert.ToDouble(values[1]));
                }
            }
        }

        private static void Main(string[] args)
        {
            // create a neural network, without using a factory
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            List<double> listA = new List<double>();
            List<double> listB = new List<double>();

            ReadCSV(ref listA, ref listB);

            double[][] arr =
            {
                listA.ToArray(),
                listB.ToArray()
            };

            networkInput = TransposeRowsColumnsExtension.TransposeRowsAndColumns(arr);
        }

    }
}
