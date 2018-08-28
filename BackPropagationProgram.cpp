// https://msdn.microsoft.com/en-us/magazine/jj658979.aspx

#include <array>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <iterator>
#include <tuple>
#include <numeric>

#include <time.h>

using std::vector;
using std::array;
using std::cout;

namespace BackPropagation
{

    enum { DIM = 28 * 28 };

    typedef unsigned char AttributeType;

    struct ObjectInfo
    {
        AttributeType pos[DIM];
        int data;
    };

    typedef std::vector<ObjectInfo> ObjectInfos;

    std::istream& operator %(std::istream& s, int32_t& v)
    {
        s.read((char*)&v, sizeof(v));
        std::reverse((char*)&v, (char*)(&v + 1));
        return s;
    }

    ObjectInfos ReadDataSet(const char* imageFile, const char* labelFile)
    {
        std::ifstream ifsImages(imageFile, std::ifstream::in | std::ifstream::binary);
        int32_t magic;
        ifsImages % magic;
        int32_t numImages;
        ifsImages % numImages;
        int32_t numRows, numCols;
        ifsImages % numRows % numCols;

        std::ifstream ifsLabels(labelFile, std::ifstream::in | std::ifstream::binary);
        ifsLabels % magic;
        int32_t numLabels;
        ifsLabels % numLabels;

        ObjectInfos infos;
        infos.resize(numImages);
        for (int i = 0; i < numImages; ++i)
        {
            ifsImages.read((char*)infos[i].pos, DIM);
            unsigned char label;
            ifsLabels.read((char*)&label, 1);
            infos[i].data = label;
        }

        const bool ok = ifsImages && ifsLabels;
        //const bool eof = ifsImages.eof() && ifsLabels.eof();

        return infos;
    }


  namespace Helpers
  {
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

    inline double SigmoidFunction(double x)
    {
        //if (x < -45.0) return 0.0;
        //else if (x > 45.0) return 1.0;
        //else return 1.0 / (1.0 + exp(-x));
        return 1.0 / (1.0 + exp(-x));
    }

  }; // namespace Helpers

  template<int numInput, int numHidden, int numOutput>
  class NeuralNetwork
  {
  private:
    int numSamples;

    //array<array<double, numHidden>, numInput> ihWeights; // input-to-hidden
    array<array<double, numInput>, numHidden> ihWeights; // input-to-hidden
    array<double, numHidden> ihBiases;

    //array<array<double, numOutput>, numHidden> hoWeights;  // hidden-to-output
    array<array<double, numHidden>, numOutput> hoWeights;  // hidden-to-output
    array<double, numOutput> hoBiases;

    array<array<double, numHidden>, numInput> ihPrevWeightsDelta;  // for momentum with back-propagation
    array<double, numHidden> ihPrevBiasesDelta;

    array<array<double, numOutput>, numHidden> hoPrevWeightsDelta;
    array<double, numOutput> hoPrevBiasesDelta;

  public: 
    NeuralNetwork()
        : numSamples(0)
    {
        for (auto& v : ihPrevWeightsDelta)
            v.fill(0);
        ihPrevBiasesDelta.fill(0);

        for (auto& v : hoPrevWeightsDelta)
            v.fill(0);
        hoPrevBiasesDelta.fill(0);
    }

    void UpdateWeights(
        const vector<double>& inputs, 
        const vector<double>& ihOutputs,
        const vector<double>& outputs,
        const vector<double>& tValues) // update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
    {
        vector<double> oGrads(numOutput); // output gradients for back-propagation
        vector<double> hGrads(numHidden); // hidden gradients for back-propagation


      // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)
      if (tValues.size() != numOutput)
        throw std::runtime_error("target values not same Length as output in UpdateWeights");

      // 1. compute output gradients
      for (int i = 0; i < oGrads.size(); ++i)
      {
        const double derivative = (1 - outputs[i]) * outputs[i]; // derivative of sigmoid
        oGrads[i] = derivative * (tValues[i] - outputs[i]);
      }

      // 2. compute hidden gradients
      for (int i = 0; i < hGrads.size(); ++i)
      {
        const double derivative = (1 - ihOutputs[i]) * ihOutputs[i]; // (1 / 1 + exp(-x))'  -- using output value of neuron
        double sum = 0.0;
        for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
          sum += oGrads[j] * hoWeights[j][i]; // each downstream gradient * outgoing weight
        hGrads[i] = derivative * sum;
      }

      // 3. update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
      for (int i = 0; i < ihPrevWeightsDelta.size(); ++i) // 0..2 (3)
      {
        for (int j = 0; j < ihPrevWeightsDelta[0].size(); ++j) // 0..3 (4)
        {
          const double delta = hGrads[j] * inputs[i]; // compute the new delta
          ihPrevWeightsDelta[i][j] += delta; // update
        }
      }

      // 3b. update input to hidden biases
      for (int i = 0; i < ihPrevBiasesDelta.size(); ++i)
      {
        const double delta = hGrads[i] * 1.0; // the 1.0 is the constant input for any bias; could leave out
        ihPrevBiasesDelta[i] += delta;
      }

      // 4. update hidden to output weights
      for (int i = 0; i < hoPrevWeightsDelta.size(); ++i)  // 0..3 (4)
      {
        for (int j = 0; j < hoPrevWeightsDelta[0].size(); ++j) // 0..1 (2)
        {
          const double delta = oGrads[j] * ihOutputs[i];  // see above: ihOutputs are inputs to next layer
          hoPrevWeightsDelta[i][j] += delta;
        }
      }

      // 4b. update hidden to output biases
      for (int i = 0; i < hoPrevBiasesDelta.size(); ++i)
      {
        const double delta = oGrads[i] * 1.0;
        hoPrevBiasesDelta[i] += delta;
      }

      ++numSamples;
    } // UpdateWeights

    void ApplyDeltas(double eta, double lambda)
    {
        if (numSamples <= 0)
            return;

        // https://jamesmccaffrey.wordpress.com/2017/02/19/l2-regularization-and-back-propagation/
        const double weightCoeff = 1. - eta * lambda / numSamples;
        const double deltaCoeff = eta / numSamples;

        // update input to hidden weights (gradients must be computed right-to-left but weights can be updated in any order
        for (int i = 0; i < ihWeights.size(); ++i) // 0..2 (3)
        {
            for (int j = 0; j < ihWeights[0].size(); ++j) // 0..3 (4)
            {
                ihWeights[i][j] = ihWeights[i][j] * weightCoeff + ihPrevWeightsDelta[j][i] * deltaCoeff; // update
                ihPrevWeightsDelta[j][i] = 0;
            }
        }

        // update input to hidden biases
        for (int i = 0; i < ihBiases.size(); ++i)
        {
            ihBiases[i] += ihPrevBiasesDelta[i] * deltaCoeff;
            ihPrevBiasesDelta[i] = 0;
        }

        // update hidden to output weights
        for (int i = 0; i < hoWeights.size(); ++i)  // 0..3 (4)
        {
            for (int j = 0; j < hoWeights[0].size(); ++j) // 0..1 (2)
            {
                hoWeights[i][j] = hoWeights[i][j] * weightCoeff + hoPrevWeightsDelta[j][i] * deltaCoeff;
                hoPrevWeightsDelta[j][i] = 0;
            }
        }

        // update hidden to output biases
        for (int i = 0; i < hoBiases.size(); ++i)
        {
            hoBiases[i] += hoPrevBiasesDelta[i] * deltaCoeff;
            hoPrevBiasesDelta[i] = 0;
        }

        numSamples = 0;
    }


    void SetRandomWeights(double epsilon_init)
    {
        std::uniform_real_distribution<double> dis(-epsilon_init, epsilon_init);
        std::default_random_engine re;

        for (int i = 0; i < ihWeights[0].size(); ++i)
            for (int j = 0; j < ihWeights.size(); ++j)
                ihWeights[j][i] = dis(re);

        for (int i = 0; i < numHidden; ++i)
            ihBiases[i] = dis(re);

        for (int i = 0; i < hoWeights[0].size(); ++i)
            for (int j = 0; j < hoWeights.size(); ++j)
                hoWeights[j][i] = dis(re);

        for (int i = 0; i < numOutput; ++i)
            hoBiases[i] = dis(re);
    }

    auto ComputeOutputs(const vector<double>& inputs)
    {
      if (inputs.size() != numInput)
      {
          std::ostringstream s;
          s << "Inputs array length " << inputs.size() << " does not match NN numInput value " << numInput;
          throw std::runtime_error(s.str());
      }

      vector<double> ihSums(numHidden);
      vector<double> hoSums(numOutput);

      std::tuple<vector<double>, vector<double>> result;
      auto& [ihOutputs, outputs] = result;
      ihOutputs.resize(numHidden);
      outputs.resize(numOutput);
      
      for (int j = 0; j < numHidden; ++j)  // compute input-to-hidden weighted sums
          ihSums[j] += std::inner_product(inputs.begin(), inputs.end(), ihWeights[j].begin(), 0.);
        //for (int i = 0; i < numInput; ++i)
        //  ihSums[j] += inputs[i] * ihWeights[j][i];

      for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
        ihSums[i] += ihBiases[i];

      for (int i = 0; i < numHidden; ++i)   // determine input-to-hidden output
        ihOutputs[i] = Helpers::SigmoidFunction(ihSums[i]);

      for (int j = 0; j < numOutput; ++j)   // compute hidden-to-output weighted sums
          hoSums[j] += std::inner_product(ihOutputs.begin(), ihOutputs.end(), hoWeights[j].begin(), 0.);
        //for (int i = 0; i < numHidden; ++i)
        //  hoSums[j] += ihOutputs[i] * hoWeights[j][i];

      for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
        hoSums[i] += hoBiases[i];

      for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
        outputs[i] = Helpers::SigmoidFunction(hoSums[i]);

      return result;
    } // ComputeOutputs

  }; // class NeuralNetwork

} // ns


int main()
{
    using namespace BackPropagation;

    try
    {
        cout << "\nBegin Neural Network Back-Propagation demo\n\n";

        //cout << "Creating a 3-input, 4-hidden, 2-output neural network\n";
        cout << "Using sigmoid function for input-to-hidden activation\n";
        cout << "Using tanh function for hidden-to-output activation\n";
        BackPropagation::NeuralNetwork<DIM, 25, 10> nn;// (DIM, 25, 10);

        nn.SetRandomWeights(0.05);

        auto trainingSet = ReadDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        //trainingSet.resize(1000);

        const double eta = 0.9;  // learning rate - controls the maginitude of the increase in the change in weights. found by trial and error.
        const double lambda = 0.01;
        //const double alpha = 0.04; // momentum - to discourage oscillation. found by trial and error.
        cout << "Setting learning rate (eta) = " << std::setprecision(2) << eta << '\n'; // << " and momentum (alpha) = " << std::setprecision(2) << alpha << '\n';

        cout << "\nEntering main back-propagation compute-update cycle\n";
        cout << "Stopping when sum absolute error <= 0.01 or 1,000 iterations\n\n";

        clock_t start = clock();

        //int ctr = 0;
        //vector<double> yValues = nn.ComputeOutputs(xValues); // prime the back-propagation loop
        //double error = BackPropagation::Helpers::Error(tValues, yValues);
        for (int ctr = 1; ctr <= 1000; ++ctr) // && error > 0.01)
        {
            //cout << "===================================================\n";
            //cout << "iteration = " << ctr << '\n';
            //cout << "Updating weights and biases using back-propagation\n";
            double error = 0;
            for (const auto& data : trainingSet)
            {
                vector<double> xValues(std::begin(data.pos), std::end(data.pos));
                for (auto& v : xValues)
                    v /= 255.;
                vector<double> tValues(10);
                tValues[data.data % 10] = 1;
                const auto [ihOutputs, yValues] = nn.ComputeOutputs(xValues);
                error += BackPropagation::Helpers::Error(tValues, yValues);
                nn.UpdateWeights(xValues, ihOutputs, yValues, tValues);
                //cout << "Computing new outputs:\n";
            }

            cout << "Iteration = " << ctr << " error = " << std::setprecision(4) << error << '\n';
            //cout << "Error = " << error << '\n';

            nn.ApplyDeltas(eta, lambda);

            //BackPropagation::Helpers::ShowVector(yValues, 4, false);
            //cout << "\nComputing new error\n";
            //error = BackPropagation::Helpers::Error(tValues, yValues);
            //cout << "Error = " << std::setprecision(4) << error << '\n';
            //++ctr;
        }
        //cout << "===================================================\n";
        //cout << "\nBest weights and biases found:\n";
        //vector<double> bestWeights = nn.GetWeights();
        //BackPropagation::Helpers::ShowVector(bestWeights, 2, true);

        cout << "  back-propagation time: " <<
            (double)(clock() - start) / CLOCKS_PER_SEC <<
            " seconds" << std::endl;

        auto testSet = ReadDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

        int numMismatches = 0;

        std::for_each(testSet.begin(), testSet.end(),
            [&nn, &numMismatches](const auto& data)
        {
            vector<double> xValues(std::begin(data.pos), std::end(data.pos));
            for (auto& v : xValues)
                v /= 255.;
            const auto[ihOutputs, yValues] = nn.ComputeOutputs(xValues);

            auto predicted = std::max_element(yValues.begin(), yValues.end()) - yValues.begin();

            if (predicted != data.data % 10)
                ++numMismatches;
        });

        cout << "Test cases: " << testSet.size() << "; mismatches: " << numMismatches << '\n';

        cout << "End Neural Network Back-Propagation demo\n\n";
        //Console.ReadLine();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        //Console.ReadLine();
    }
} // Main
