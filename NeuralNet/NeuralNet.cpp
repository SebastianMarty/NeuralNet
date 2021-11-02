#include "Neuron.h"
#include <vector>
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <bitset>

using namespace std;

int hiddenLayersCount = 10;
int hiddenNeuronsCount = 20;
int trainingRounds = 100;

double learningRate = 1;
double avgError = 0.0f;
double recentAvgError = 0.0f;
double recentAvgSmoothingFactor = 100.0f;

vector<double> inputs;
vector<double> biases;
vector<double> expOutputs;
vector<vector<Neuron*>> neurons;
vector<Neuron*> inputNeurons;
vector<Neuron*> outputNeurons;

double Randf()
{
	return rand() / double(RAND_MAX);
}

int GetNeuronIndex(Neuron* neuron, int layerIndex)
{
	auto it = find(neurons[layerIndex].begin(), neurons[layerIndex].end(), neuron);
	return it - neurons[layerIndex].begin();
}

int GetLayerIndex(vector<Neuron*> layer)
{
	auto it = find(neurons.begin(), neurons.end(), layer);
	return it - neurons.begin();
}

void GetInputsFromImage(string path)
{
	cv::Mat image = cv::imread(path);

	if (!image.empty())
	{
		//Resize image to fixed input size
		cv::Mat imgResized;
		double scaleFactor = 50 / (double)image.size().width;
		cv::resize(image, imgResized, cv::Size(image.size().width * scaleFactor, image.size().height * scaleFactor), cv::InterpolationFlags::INTER_LINEAR);

		inputs = {};

		// Iterate over all pixels of the image
		for (int r = 0; r < imgResized.rows; r++) {
			// Obtain a pointer to the beginning of row r
			cv::Vec3b* ptr = imgResized.ptr<cv::Vec3b>(r);

			for (int c = 0; c < imgResized.cols; c++) {
				// Get average (grayscale) value of pixel and divide by 255 to normalize the pixel value
				int i = (ptr[c][0] + ptr[c][1] + ptr[c][2]) / 3.0f;
				inputs.push_back(((double)i / (double)255) * -1.0f + 1.0f);
			}
		}
	}
}

void InitializeNet()
{
	for (double input : inputs)
	{
		inputNeurons.push_back(new Neuron(input));
	}

	biases.push_back(Randf());

	neurons.push_back(inputNeurons);

	for (int x = 0; x < hiddenLayersCount; x++)
	{
		vector<Neuron*> hiddenNeurons;

		for (int y = 0; y < hiddenNeuronsCount; y++)
		{
			double value = 0.0f;

			for (Neuron* prevNeuron : neurons[x])
			{
				prevNeuron->AddWeight(Randf());
				value += prevNeuron->GetValue() * prevNeuron->GetWeightAt(y);
			}

			value += biases[x];
			Neuron* neuron = new Neuron(value);
			neuron->Activate();
			hiddenNeurons.push_back(neuron);
		}

		biases.push_back(Randf());

		neurons.push_back(hiddenNeurons);
	}

	for (int x = 0; x < expOutputs.size(); x++)
	{
		double value = 0.0f;

		for (Neuron* prevNeuron : neurons.back())
		{
			prevNeuron->AddWeight(Randf());
			value += prevNeuron->GetValue() * prevNeuron->GetWeightAt(x);
		}

		value += biases.back();
		Neuron* neuron = new Neuron(value);
		neuron->Activate();
		outputNeurons.push_back(neuron);
	}

	biases.push_back(Randf());

	neurons.push_back(outputNeurons);
}

void FeedForward()
{
	for (int x = 1; x < neurons.size(); x++)
	{
		for (int y = 0; y < neurons[x].size(); y++)
		{
			double value = 0.0f;

			for (Neuron* prevNeuron : neurons[x - 1])
			{
				value += prevNeuron->GetWeightAt(y) * prevNeuron->GetValue();
			}

			value += biases[x];

			neurons[x][y]->SetValue(value);
			neurons[x][y]->Activate();
		}
	}
}

void BackPropagate()
{
	vector<Neuron*> outputLayer = neurons.back();
	avgError = 0.0f;

	for (int x = 0; x < outputLayer.size(); x++)
	{
		double delta = expOutputs[x] - outputLayer[x]->GetValue();
		avgError += delta * delta;
	}

	avgError /= outputLayer.size(); // get average error squared
	avgError = sqrt(avgError); // root mean squared error (rmse)

	// recent average measurement
	recentAvgError = (recentAvgError * recentAvgSmoothingFactor + avgError) / (recentAvgSmoothingFactor + 1.0f);

	// calculate output layer gradients
	for (int x = 0; x < outputLayer.size(); x++)
	{
		outputLayer[x]->CalcOutputGradients(expOutputs[x]);
	}

	// calculate gradients on hidden layers
	for (int x = neurons.size() - 2; x >= 0; x--)
	{
		vector<Neuron*> hiddenLayer = neurons[x];
		vector<Neuron*> nextLayer = neurons[x + 1];

		for (int y = 0; y < hiddenLayer.size(); y++)
		{
			hiddenLayer[y]->CalcHiddenGradients(nextLayer);
		}
	}

	// for all layers from outputs to first hidden layer, update connection weights
	for (int x = neurons.size() - 1; x > 0; x--)
	{
		vector<Neuron*> layer = neurons[x];
		vector<Neuron*> prevLayer = neurons[x - 1];

		for (int y = 0; y < layer.size() - 1; y++)
		{
			int layerIndex = GetLayerIndex(layer);
			int neuronIndex = GetNeuronIndex(layer[y], layerIndex);
			layer[y]->UpdateWeights(prevLayer, neuronIndex, learningRate);
		}
	}
}

void WriteAll(char sample)
{
	std::string path("D:\\Smartlearn\\5_daten\\4. Lehrjahr\\1. Semester\\ABU\\VA\\NeuralNet\\Results\\Sample_");
	path.push_back(sample);
	path.append(".txt");

	fstream file(path, ios_base::app | ios_base::out | ios_base::in);

	for (int x = 1; x < neurons.size(); x++)
	{
		auto it = find(neurons.begin(), neurons.end(), neurons[x]);
		int index = it - neurons.begin();

		file << "Layer: " << index + 1 << std::endl;
		file << "Layer Bias: ";
		file << biases[index] << std::endl;
		file << "***************************" << std::endl;

		for (int y = 0; y < neurons[x].size(); y++)
		{
			file << "Neuron Value: " << neurons[x][y]->GetValue() << std::endl;

			file << "Neuron Weights:" << std::endl;

			for (double weight : neurons[x][y]->GetAllWeights())
			{
				file << weight << std::endl;
			}

			file << "+++++++++++++++++++++++++++" << std::endl;
			file << std::endl;
		}

		file << std::endl;
		file << std::endl;
	}

	file.close();
}

int main(int argc, char* argv[])
{
	if (argv[1])
	{
		string text;
		ifstream stream(argv[1]);
		int runCount = 0;
		char lastExpResult = '0';

		while (getline(stream, text))
		{
			if (runCount == 0)
			{
				runCount++;
				continue;
			}

			string path;
			char expResult = '0';

			for (int x = 0; x < text.length(); x++)
			{
				if (text[x] == ',')
				{
					expResult = text[x + 2];
					break;
				}
				else
				{
					if (text[x] != '\0')
					{
						path += text[x];
					}
				}
			}

			if (lastExpResult != expResult)
			{
				WriteAll(lastExpResult);
				lastExpResult = expResult;
			}

			vector<double> expValues;

			for (int c : bitset<8>(expResult).to_string())
			{
				expValues.push_back(c - 48);
			}

			expOutputs = expValues;
			expValues = {};

			cout << bitset<8>(expResult).to_string() << "|" << expResult << endl;

			GetInputsFromImage(argv[2] + path);

			if (neurons.size() == 0)
			{
				InitializeNet();
				BackPropagate();
			}

			// Training the network
			for (int x = 0; x < trainingRounds; x++)
			{
				FeedForward();
				BackPropagate();

				/*cout << "Training round: " << x + 1 << endl;

				for (Neuron* neuron : neurons.back())
				{
					cout << neuron->GetValue() << endl;
				}*/

			}
		}
	}
}