using System;
using System.Collections.Generic;
using System.IO;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using Encog.Util.Arrayutil;

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
        public static double[][] networkExpectedOut;

        private static void ReadInputCSV(ref List<double> listA, ref List<double> listB)
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

        private static void ReadExpectedOutCSV(ref List<double> listA, ref List<double> listB)
        {
            using (var reader = new StreamReader(@".\homeloan.csv"))
            {
                listA = new List<double>();
                listB = new List<double>();
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');

                    listA.Add(Convert.ToDouble(values[0]));
                    //listB.Add(Convert.ToDouble(values[1]));
                }
            }
        }


        private static void Main(string[] args)
        {
            // Read and Normalise Input CSV to array
            List<double> listA = new List<double>();
            List<double> listB = new List<double>();

            ReadInputCSV(ref listA, ref listB);

            double[][] arrIn =
            {
                listA.ToArray(),
                listB.ToArray()
            };

            networkInput = TransposeRowsColumnsExtension.TransposeRowsAndColumns(arrIn);

            var normIncome = new NormalizedField(NormalizationAction.Normalize, "Income", 20000000, 100000, 1, 0);
            var normEduExpense = new NormalizedField(NormalizationAction.Normalize, "Edu Expense", 3000, 0, 1, 0);

            for (int i = 0; i < networkInput.Length; i++)
            {
                networkInput[i][0] = normIncome.Normalize(networkInput[i][0]);
                networkInput[i][1] = normEduExpense.Normalize(networkInput[i][1]);

            }

            // Read and Normalise ExpectedOutput CSV to array
            listA = null;
            listB = null;
            ReadExpectedOutCSV(ref listA, ref listB);

            double [][] arrExpectedOut = 
            {
                listA.ToArray(),
                //listB.ToArray()
            };

            networkExpectedOut = TransposeRowsColumnsExtension.TransposeRowsAndColumns(arrExpectedOut);

//            var normHomeLoanInt = new NormalizedField(NormalizationAction.Normalize, "Home Loan Interest", 50000, 0, 1, 0);
            var normHomeLoanInt = new NormalizedField(NormalizationAction.Normalize, "Home Loan Interest", 10000, 0, 1, 0);

            for (int i = 0; i < networkInput.Length; i++)
            {
                networkExpectedOut[i][0] = normHomeLoanInt.Normalize(networkExpectedOut[i][0]);

            }

            // create a neural network, without using a factory
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            // create training data
            IMLDataSet trainingSet = new BasicMLDataSet(networkInput, networkExpectedOut);

            // train the neural network
            IMLTrain train = new ResilientPropagation(network, trainingSet);

            int epoch = 1;

            do
            {
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" Error:" + train.Error);
                epoch++;
            } while (train.Error > 0.005);

            train.FinishTraining();

            // test the neural network
            double inputIncome = 250000;
            double inputEduExpense = 500;
            inputIncome = normIncome.Normalize(inputIncome);
            inputEduExpense = normEduExpense.Normalize(inputEduExpense);

            double[] runningInput = { inputIncome, inputEduExpense };

           double[] runningOutput = { 0.0 };

            Console.WriteLine(@"Neural Network Results:");
            network.Compute(runningInput, runningOutput);
            //            Console.WriteLine("Input = " + inputIncome.ToString + @" and " & inputEduExpense.ToString);
            Console.WriteLine(String.Concat("Input = ", normIncome.DeNormalize(inputIncome), @" and ", normEduExpense.DeNormalize(inputEduExpense)));
            Console.WriteLine(String.Concat("Output = ", normHomeLoanInt.DeNormalize(runningOutput[0])));


            EncogFramework.Instance.Shutdown();
        }
    }
}
