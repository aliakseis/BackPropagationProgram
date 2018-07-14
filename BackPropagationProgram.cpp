// https://msdn.microsoft.com/en-us/magazine/jj658979.aspx

#include <vector>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

using std::vector;
using std::cout;

namespace BackPropagation
{

  namespace Helpers
  {

    vector<vector<double>> MakeMatrix(int rows, int cols)
    {
      vector<vector<double>> result(rows);
      for (int i = 0; i < rows; ++i)
        result[i].resize(cols);
      return result;
    }

    void ShowVector(const vector<double>& vector, int decimals, bool blankLine)
    {
      for (int i = 0; i < vector.size(); ++i)
      {
        if (i > 0 && i % 12 == 0) // max of 12 values per row 
          cout << '\n';
        if (vector[i] >= 0.0) 
            cout << ' ';
        cout << std::fixed << std::setprecision(decimals) << vector[i] << ' '; // n decimals
      }
      if (blankLine) 
          cout << "\n\n";
    }

    void ShowMatrix(const vector<vector<double>>& matrix, int numRows, int decimals)
    {
      int ct = 0;
      if (numRows == -1) 
          numRows = INT_MAX; // if numRows == -1, show all rows
      for (int i = 0; i < matrix.size() && ct < numRows; ++i)
      {
        for (int j = 0; j < matrix[0].size(); ++j)
        {
          if (matrix[i][j] >= 0.0) 
              cout << ' '; // blank space instead of '+' sign
          cout << std::fixed << std::setprecision(decimals) << matrix[i][j] << ' ';
        }
        cout << '\n';
        ++ct;
      }
      cout << '\n';
    }

    double Error(const vector<double>& target, const vector<double>& output) // sum absolute error. could put into NeuralNetwork class.
    {
        double sum = 0.0;
        for (int i = 0; i < target.size(); ++i)
            sum += abs(target[i] - output[i]);
        return sum;
    }

  }; // namespace Helpers

  class NeuralNetwork
  {
  private:
    int numInput;
    int numHidden;
    int numOutput;

    vector<double> inputs;
    vector<vector<double>> ihWeights; // input-to-hidden
    vector<double> ihSums;
    vector<double> ihBiases;
    vector<double> ihOutputs;

    vector<vector<double>> hoWeights;  // hidden-to-output
    vector<double> hoSums;
    vector<double> hoBiases;
    vector<double> outputs;

    vector<double> oGrads; // output gradients for back-propagation
    vector<double> hGrads; // hidden gradients for back-propagation

    vector<vector<double>> ihPrevWeightsDelta;  // for momentum with back-propagation
    vector<double> ihPrevBiasesDelta;

    vector<vector<double>> hoPrevWeightsDelta;
    vector<double> hoPrevBiasesDelta;

  public: 
    NeuralNetwork(int numInput, int numHidden, int numOutput)
        : numInput(numInput)
        , numHidden(numHidden)
        , numOutput(numOutput)
    {

      inputs.resize(numInput);
      ihWeights = Helpers::MakeMatrix(numInput, numHidden);
      ihSums.resize(numHidden);
      ihBiases.resize(numHidden);
      ihOutputs.resize(numHidden);
      hoWeights = Helpers::MakeMatrix(numHidden, numOutput);
      hoSums.resize(numOutput);
      hoBiases.resize(numOutput);
      outputs.resize(numOutput);

      oGrads.resize(numOutput);
      hGrads.resize(numHidden);

      ihPrevWeightsDelta = Helpers::MakeMatrix(numInput, numHidden);
      ihPrevBiasesDelta.resize(numHidden);
      hoPrevWeightsDelta = Helpers::MakeMatrix(numHidden, numOutput);
      hoPrevBiasesDelta.resize(numOutput);
    }

    void UpdateWeights(const vector<double>& tValues, double eta, double alpha) // update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
    {
      // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)
      if (tValues.size() != numOutput)
        throw std::runtime_error("target values not same Length as output in UpdateWeights");

      // 1. compute output gradients
      for (int i = 0; i < oGrads.size(); ++i)
      {
        double derivative = (1 - outputs[i]) * (1 + outputs[i]); // derivative of tanh
        oGrads[i] = derivative * (tValues[i] - outputs[i]);
      }

      // 2. compute hidden gradients
      for (int i = 0; i < hGrads.size(); ++i)
      {
        double derivative = (1 - ihOutputs[i]) * ihOutputs[i]; // (1 / 1 + exp(-x))'  -- using output value of neuron
        double sum = 0.0;
        for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
          sum += oGrads[j] * hoWeights[i][j]; // each downstream gradient * outgoing weight
        hGrads[i] = derivative * sum;
      }

      // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
      for (int i = 0; i < ihWeights.size(); ++i) // 0..2 (3)
      {
        for (int j = 0; j < ihWeights[0].size(); ++j) // 0..3 (4)
        {
          double delta = eta * hGrads[j] * inputs[i]; // compute the new delta
          ihWeights[i][j] += delta; // update
          ihWeights[i][j] += alpha * ihPrevWeightsDelta[i][j]; // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
        }
      }

      // 3b. update input to hidden biases
      for (int i = 0; i < ihBiases.size(); ++i)
      {
        double delta = eta * hGrads[i] * 1.0; // the 1.0 is the constant input for any bias; could leave out
        ihBiases[i] += delta;
        ihBiases[i] += alpha * ihPrevBiasesDelta[i];
      }

      // 4. update hidden to output weights
      for (int i = 0; i < hoWeights.size(); ++i)  // 0..3 (4)
      {
        for (int j = 0; j < hoWeights[0].size(); ++j) // 0..1 (2)
        {
          double delta = eta * oGrads[j] * ihOutputs[i];  // see above: ihOutputs are inputs to next layer
          hoWeights[i][j] += delta;
          hoWeights[i][j] += alpha * hoPrevWeightsDelta[i][j];
          hoPrevWeightsDelta[i][j] = delta;
        }
      }

      // 4b. update hidden to output biases
      for (int i = 0; i < hoBiases.size(); ++i)
      {
        double delta = eta * oGrads[i] * 1.0;
        hoBiases[i] += delta;
        hoBiases[i] += alpha * hoPrevBiasesDelta[i];
        hoPrevBiasesDelta[i] = delta;
      }
    } // UpdateWeights

    void SetWeights(const vector<double>& weights)
    {
      // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      if (weights.size() != numWeights)
      {
          std::ostringstream s;
          s << "The weights array length: " << weights.size() << " does not match the total number of weights and biases: " << numWeights;
          throw std::runtime_error(s.str());
      }

      int k = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];

      for (int i = 0; i < numHidden; ++i)
        ihBiases[i] = weights[k++];

      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];

      for (int i = 0; i < numOutput; ++i)
        hoBiases[i] = weights[k++];
    }

    vector<double> GetWeights()
    {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      vector<double> result(numWeights);
      int k = 0;
      for (int i = 0; i < ihWeights.size(); ++i)
        for (int j = 0; j < ihWeights[0].size(); ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < ihBiases.size(); ++i)
        result[k++] = ihBiases[i];
      for (int i = 0; i < hoWeights.size(); ++i)
        for (int j = 0; j < hoWeights[0].size(); ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < hoBiases.size(); ++i)
        result[k++] = hoBiases[i];
      return result;
    }

    vector<double> ComputeOutputs(const vector<double>& xValues)
    {
      if (xValues.size() != numInput)
      {
          std::ostringstream s;
          s << "Inputs array length " << inputs.size() << " does not match NN numInput value " << numInput;
          throw std::runtime_error(s.str());
      }

      for (int i = 0; i < numHidden; ++i)
        ihSums[i] = 0.0;
      for (int i = 0; i < numOutput; ++i)
        hoSums[i] = 0.0;

      for (int i = 0; i < xValues.size(); ++i) // copy x-values to inputs
        inputs[i] = xValues[i];
      
      for (int j = 0; j < numHidden; ++j)  // compute input-to-hidden weighted sums
        for (int i = 0; i < numInput; ++i)
          ihSums[j] += inputs[i] * ihWeights[i][j];

      for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
        ihSums[i] += ihBiases[i];

      for (int i = 0; i < numHidden; ++i)   // determine input-to-hidden output
        ihOutputs[i] = SigmoidFunction(ihSums[i]);

      for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output weighted sums
        for (int i = 0; i < numHidden; ++i)
          hoSums[j] += ihOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
        hoSums[i] += hoBiases[i];

      for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
        outputs[i] = HyperTanFunction(hoSums[i]);

      return outputs;
    } // ComputeOutputs

    private:
    static double StepFunction(double x) // an activation function that isn't compatible with back-propagation bcause it isn't differentiable
    {
      if (x > 0.0) return 1.0;
      else return 0.0;
    }

    static double SigmoidFunction(double x)
    {
      if (x < -45.0) return 0.0;
      else if (x > 45.0) return 1.0;
      else return 1.0 / (1.0 + exp(-x));
    }

    static double HyperTanFunction(double x)
    {
      if (x < -10.0) return -1.0;
      else if (x > 10.0) return 1.0;
      else return tanh(x);
    }
  }; // class NeuralNetwork

} // ns


int main()
{
    try
    {
        cout << "\nBegin Neural Network Back-Propagation demo\n\n";

        cout << "Creating a 3-input, 4-hidden, 2-output neural network\n";
        cout << "Using sigmoid function for input-to-hidden activation\n";
        cout << "Using tanh function for hidden-to-output activation\n";
        BackPropagation::NeuralNetwork nn(3, 4, 2);

        // arbitrary weights and biases
        vector<double> weights {
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                -2.0, -6.0, -1.0, -7.0,
                1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                -2.5, -5.0 };

        cout << "\nInitial 26 random weights and biases are:\n";
        BackPropagation::Helpers::ShowVector(weights, 2, true);

        cout << "Loading neural network weights and biases\n";
        nn.SetWeights(weights);

        cout << "\nSetting inputs:\n";
        vector<double> xValues { 1.0, 2.0, 3.0 };
        BackPropagation::Helpers::ShowVector(xValues, 2, true);

        vector<double> initialOutputs = nn.ComputeOutputs(xValues);
        cout << "Initial outputs:\n";
        BackPropagation::Helpers::ShowVector(initialOutputs, 4, true);

        vector<double> tValues { -0.8500, 0.7500 }; // target (desired) values. note these only make sense for tanh output activation
        cout << "Target outputs to learn are:\n";
        BackPropagation::Helpers::ShowVector(tValues, 4, true);

        double eta = 0.90;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
        double alpha = 0.04; // momentum - to discourage oscillation. found by trial and error.
        cout << "Setting learning rate (eta) = " << std::setprecision(2) << eta << " and momentum (alpha) = " << std::setprecision(2) << alpha << '\n';

        cout << "\nEntering main back-propagation compute-update cycle\n";
        cout << "Stopping when sum absolute error <= 0.01 or 1,000 iterations\n\n";
        int ctr = 0;
        vector<double> yValues = nn.ComputeOutputs(xValues); // prime the back-propagation loop
        double error = BackPropagation::Helpers::Error(tValues, yValues);
        while (ctr < 1000 && error > 0.01)
        {
            cout << "===================================================\n";
            cout << "iteration = " << ctr << '\n';
            cout << "Updating weights and biases using back-propagation\n";
            nn.UpdateWeights(tValues, eta, alpha);
            cout << "Computing new outputs:\n";
            yValues = nn.ComputeOutputs(xValues);
            BackPropagation::Helpers::ShowVector(yValues, 4, false);
            cout << "\nComputing new error\n";
            error = BackPropagation::Helpers::Error(tValues, yValues);
            cout << "Error = " << std::setprecision(4) << error << '\n';
            ++ctr;
        }
        cout << "===================================================\n";
        cout << "\nBest weights and biases found:\n";
        vector<double> bestWeights = nn.GetWeights();
        BackPropagation::Helpers::ShowVector(bestWeights, 2, true);

        cout << "End Neural Network Back-Propagation demo\n\n";
        //Console.ReadLine();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        //Console.ReadLine();
    }
} // Main
