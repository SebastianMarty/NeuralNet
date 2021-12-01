#include <iostream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;

class Neuron;
typedef std::vector<Neuron> Layer;

//int hiddenLayersCount = 1;
//int hiddenNeuronsCount = 512;
int trainingRounds = 40;
//
//double learningRate = 0.2f;
//double error = 0.0f;
//double averageError = 0.0f;
//double recentAvgSmoothingFactor = 100.0f;
//
vector<double> inputs;
//vector<double> biases;
vector<double> expOutputs;
//vector<vector<Neuron*>> neurons;
//vector<Neuron*> inputNeurons;
//vector<Neuron*> outputNeurons;

//double Randf()
//{
//	return (double)rand() / double(RAND_MAX);
//}
//
//int GetNeuronIndex(Neuron* neuron, int layerIndex)
//{
//	auto it = find(neurons[layerIndex].begin(), neurons[layerIndex].end(), neuron);
//	return it - neurons[layerIndex].begin();
//}
//
//int GetLayerIndex(vector<Neuron*> layer)
//{
//	auto it = find(neurons.begin(), neurons.end(), layer);
//	return it - neurons.begin();
//}

void GetInputsFromImage(string path)
{
	cv::Mat image = cv::imread(path);

	if (!image.empty())
	{
		//Resize image to fixed input size
		cv::Mat imgResized;
		cv::resize(image, imgResized, cv::Size(32, 32), cv::InterpolationFlags::INTER_LINEAR);

		//cv::imshow("Test", imgResized);
		//cv::waitKey();

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

//void InitializeNet()
//{
//	for (double input : inputs)
//	{
//		inputNeurons.push_back(new Neuron(input));
//	}
//
//	biases.push_back(Randf());
//
//	neurons.push_back(inputNeurons);
//
//	for (int x = 0; x < hiddenLayersCount; x++)
//	{
//		vector<Neuron*> hiddenNeurons;
//
//		for (int y = 0; y < hiddenNeuronsCount; y++)
//		{
//			double value = 0.0f;
//
//			for (Neuron* prevNeuron : neurons[x])
//			{
//				prevNeuron->AddWeight(Randf());
//				value += prevNeuron->GetValue() * prevNeuron->GetWeightAt(y);
//			}
//
//			value += biases[x];
//			Neuron* neuron = new Neuron(value);
//			neuron->Activate();
//			hiddenNeurons.push_back(neuron);
//		}
//
//		biases.push_back(Randf());
//
//		neurons.push_back(hiddenNeurons);
//	}
//
//	for (int x = 0; x < expOutputs.size(); x++)
//	{
//		double value = 0.0f;
//
//		for (Neuron* prevNeuron : neurons.back())
//		{
//			prevNeuron->AddWeight(Randf());
//			value += prevNeuron->GetValue() * prevNeuron->GetWeightAt(x);
//		}
//
//		value += biases.back();
//		Neuron* neuron = new Neuron(value);
//		neuron->Activate();
//		outputNeurons.push_back(neuron);
//	}
//
//	biases.push_back(Randf());
//
//	neurons.push_back(outputNeurons);
//}
//
//void FeedForward()
//{
//	for (int x = 1; x < neurons.size(); x++)
//	{
//		for (int y = 0; y < neurons[x].size(); y++)
//		{
//			double value = 0.0f;
//
//			for (Neuron* prevNeuron : neurons[x - 1])
//			{
//				value += prevNeuron->GetWeightAt(y) * prevNeuron->GetValue();
//			}
//
//			value += biases[x];
//
//			neurons[x][y]->SetValue(value);
//			neurons[x][y]->Activate();
//		}
//	}
//}
//
//void BackPropagate()
//{
//	vector<Neuron*> outputLayer = neurons.back();
//	error = 0.0f;
//
//	for (int x = 0; x < outputLayer.size(); x++)
//	{
//		double delta = expOutputs[x] - (outputLayer[x]->GetValue());
//		error += delta * delta;
//	}
//
//	error /= outputLayer.size() - 1; // get average error squared
//	error = sqrt(error); // root mean squared error (rmse)
//
//	// recent average measurement
//	averageError = (averageError * recentAvgSmoothingFactor + error) / (recentAvgSmoothingFactor + 1.0f);
//
//	// calculate output layer gradients
//	for (int x = 0; x < outputLayer.size(); x++)
//	{
//		outputLayer[x]->CalcOutputGradients(expOutputs[x]);
//	}
//
//	// calculate gradients on hidden layers
//	for (int x = neurons.size() - 2; x >= 0; x--)
//	{
//		vector<Neuron*> hiddenLayer = neurons[x];
//		vector<Neuron*> nextLayer = neurons[x + 1];
//
//		for (int y = 0; y < hiddenLayer.size(); y++)
//		{
//			hiddenLayer[y]->CalcHiddenGradients(nextLayer);
//		}
//	}
//
//	// for all layers from outputs to first hidden layer, update connection weights
//	for (int x = neurons.size() - 1; x > 0; x--)
//	{
//		vector<Neuron*> layer = neurons[x];
//		vector<Neuron*> prevLayer = neurons[x - 1];
//
//		for (int y = 0; y < layer.size() - 1; y++)
//		{
//			int layerIndex = GetLayerIndex(layer);
//			int neuronIndex = GetNeuronIndex(layer[y], layerIndex);
//			layer[y]->UpdateWeights(prevLayer, neuronIndex, learningRate);
//		}
//	}
//}










struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {

private:
	static double learningRate;
	static double alpha;
	//static double activate(double value) { return tanh(value); }
	static double activate(double value) { return 1 / (1 + exp(-value)); }
	//static double activateDerivative(double value) { return 1 - tanh(value) * tanh(value); }
	static double activateDerivative(double value) { return activate(value) * (1 - activate(value)); }
	static double random(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer& nextLayer) const;
	double output;
	std::vector<Connection> outputWeights;
	unsigned index;
	double gradient;

public:
	Neuron(unsigned outputAmt, unsigned index);
	void setOutput(double value) { output = value; }
	double getOutput(void) const { return output; }
	std::vector<Connection> getOutputWeights() const { return outputWeights; }
	void feedForward(const Layer& prevLayer);
	void calculateOutputGradients(double target);
	void calculateHiddenGradients(const Layer& nextLayer);
	void updateWeights(Layer& prevLayer);

};

double Neuron::learningRate = 0.2;
double Neuron::alpha = 0.5;


void Neuron::updateWeights(Layer& prevLayer) {
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		double oldDeltaWeight = prevLayer[n].outputWeights[index].deltaWeight;

		double newDeltaWeight = learningRate * prevLayer[n].getOutput() * gradient + alpha * oldDeltaWeight;

		prevLayer[n].outputWeights[index].deltaWeight = newDeltaWeight;
		prevLayer[n].outputWeights[index].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer& nextLayer) const {
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
		sum += outputWeights[n].weight * nextLayer[n].gradient;

	return sum;
}

void Neuron::calculateHiddenGradients(const Layer& nextLayer) {
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activateDerivative(output);
}

void Neuron::calculateOutputGradients(double target) {
	double delta = target - output;
	gradient = delta * Neuron::activateDerivative(output);
}

void Neuron::feedForward(const Layer& prevLayer) {
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); n++)
		sum += prevLayer[n].getOutput() * prevLayer[n].outputWeights[index].weight;

	output = Neuron::activate(sum);
}

Neuron::Neuron(unsigned outputAmt, unsigned index) {
	this->index = index;
	outputWeights.reserve(outputAmt);

	for (unsigned i = 0; i < outputAmt; i++) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = random();
	}
}


class Network {

private:
	std::vector<Layer> layers;
	double error;
	double averageError;
	static double smoothingFactor;

public:
	Network(const std::vector<unsigned>& layout);
	void feedForward(const std::vector<double>& inputs);
	void backProp(const std::vector<double>& targets);
	void getResults(std::vector<double>& results) const;
	double getRecentAverageError(void) const { return averageError; }
	std::vector<Layer> getLayers() const { return layers; }

};


double Network::smoothingFactor = 100;


void Network::getResults(std::vector<double>& results) const {
	results.clear();
	results.reserve(layers.back().size());

	for (unsigned n = 0; n < layers.back().size() - 1; n++) {
		results.push_back(layers.back()[n].getOutput());
	}
}

void Network::backProp(const std::vector<double>& targets) {
	Layer& outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targets[n] - outputLayer[n].getOutput();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;
	error = sqrt(error);

	averageError = (averageError * smoothingFactor + error) / (smoothingFactor + 1.0);

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calculateOutputGradients(targets[n]);
	}

	for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
		for (unsigned n = 0; n < layers[layerNum].size(); n++) {
			layers[layerNum][n].calculateHiddenGradients(layers[layerNum + 1]);
		}
	}

	for (unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
			layers[layerNum][n].updateWeights(layers[layerNum - 1]);
		}
	}
}

void Network::feedForward(const std::vector<double>& inputVals) {
	for (unsigned i = 0; i < inputVals.size(); i++) {
		layers[0][i].setOutput(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++) {
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++)
			layers[layerNum][n].feedForward(layers[layerNum - 1]);
	}
}

Network::Network(const std::vector<unsigned>& layout) {
	layers.reserve(layout.size());
	error = 0;
	averageError = 0;
	for (unsigned l = 0; l < layout.size(); l++) {
		layers.push_back(Layer());
		unsigned outputAmt = l == layout.size() - 1 ? 0 : layout[l + 1];

		layers.back().reserve(layout[l]);
		for (unsigned n = 0; n <= layout[l]; n++)
			layers.back().push_back(Neuron(outputAmt, n));

		layers.back().back().setOutput(1);
	}
}








int main(int argc, char* argv[])
{
	if (argv[1])
	{
		srand(time(NULL));

		vector<unsigned> layout{ 1024, 512, 10 };

		Network net(layout);

		for (int x = 0; x < trainingRounds; x++)
		{
			string text;
			ifstream stream(argv[1]);
			int runCount = 0;

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

				/*vector<double> expValues;

				for (int c : bitset<8>(expResult).to_string())
				{
					expValues.push_back(c - 48);
				}*/

				//expOutputs = expValues;
				//expValues = {};

				switch (expResult)
				{
				case '0':
					expOutputs = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '1':
					expOutputs = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '2':
					expOutputs = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '3':
					expOutputs = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '4':
					expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '5':
					expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '6':
					expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
					break;
				case '7':
					expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
					break;
				case '8':
					expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
					break;
				case '9':
					expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
					break;
				}

				std::cout << "Iteration: " << x + 1 << std::endl;

				GetInputsFromImage(argv[2] + path);
				net.feedForward(inputs);

				std::cout << "Inputs: " << std::flush;

				for (unsigned i = 0; i < inputs.size(); i++)
					std::cout << inputs[i] << " " << std::flush;

				net.backProp(expOutputs);

				std::cout << std::endl << "Targets: " << std::flush;

				for (unsigned i = 0; i < expOutputs.size(); i++)
					std::cout << expOutputs[i] << " " << std::flush;

				std::cout << std::endl << "Results: " << std::flush;

				std::vector<double> results;
				net.getResults(results);

				for (unsigned i = 0; i < results.size(); i++)
					std::cout << results[i] << " " << std::flush;

				std::cout << std::endl << "Average recent error: " << net.getRecentAverageError() << std::endl << std::endl;

				if (x + 1 == trainingRounds) {
					std::string choice;
					std::cout << "Neural network has reached expected amount of iterations." << std::endl;
					std::cout << "Enter Y to test the neural network, enter N to keep training it" << std::endl;
					std::cin >> choice;
					std::cin.ignore(256, '\n');
					if (choice == "Y" || choice == "y")
					{
						string testText;
						ifstream testStream(argv[1]);
						int runCount = 0;

						while (getline(testStream, testText))
						{
							if (runCount == 0)
							{
								runCount++;
								continue;
							}

							string testPath;
							char expResult = '0';

							for (int x = 0; x < testText.length(); x++)
							{
								if (testText[x] == ',')
								{
									expResult = testText[x + 2];
									break;
								}
								else
								{
									if (testText[x] != '\0')
									{
										testPath += testText[x];
									}
								}
							}

							GetInputsFromImage(testPath);
							net.feedForward(inputs);

							std::vector<double> results;
							net.getResults(results);

							int guess = 0;

							for (unsigned i = 1; i < results.size(); i++)
							{
								if (results[i] > results[guess])
								{
									guess = i;
								}
							}

							std::cout << "Guess: " << guess << std::endl;
							std::cout << "Expected: " << expResult << std::endl;
							std::cout << std::endl;
						}
					}
				}
			}
		}
















		//// Training the network
		//for (int x = 0; x < trainingRounds; x++)
		//{

		//	cout << "Training rount: " << x + 1 << endl;

		//	string text;
		//	ifstream stream(argv[1]);
		//	int runCount = 0;

		//	while (getline(stream, text))
		//	{
		//		if (runCount == 0)
		//		{
		//			runCount++;
		//			continue;
		//		}

		//		string path;
		//		char expResult = '0';

		//		for (int x = 0; x < text.length(); x++)
		//		{
		//			if (text[x] == ',')
		//			{
		//				expResult = text[x + 2];
		//				break;
		//			}
		//			else
		//			{
		//				if (text[x] != '\0')
		//				{
		//					path += text[x];
		//				}
		//			}
		//		}

		//		/*vector<double> expValues;

		//		for (int c : bitset<8>(expResult).to_string())
		//		{
		//			expValues.push_back(c - 48);
		//		}*/

		//		//expOutputs = expValues;
		//		//expValues = {};

		//		switch (expResult)
		//		{
		//		case '0':
		//			expOutputs = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '1':
		//			expOutputs = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '2':
		//			expOutputs = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '3':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '4':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '5':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '6':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
		//			break;
		//		case '7':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
		//			break;
		//		case '8':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
		//			break;
		//		case '9':
		//			expOutputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		//			break;
		//		}

		//		cout << bitset<8>(expResult).to_string() << "|" << expResult << endl;

		//		GetInputsFromImage(argv[2] + path);

		//		if (neurons.size() == 0)
		//		{
		//			InitializeNet();
		//		}
		//		else
		//		{
		//			FeedForward();
		//		}

		//		BackPropagate();

		//		std::string logPath("D:\\Smartlearn\\5_daten\\4. Lehrjahr\\1. Semester\\ABU\\VA\\NeuralNet\\Results\\TrainingResults.txt");

		//		fstream file(logPath, ios_base::app | ios_base::out | ios_base::in);
		//		file << "Training Round: " << x + 1 << endl;

		//		for (Neuron* neuron : neurons.back())
		//		{
		//			file << "Output Value: " << neuron->GetValue() << endl;
		//			file << "Expected Value: " << expOutputs[GetNeuronIndex(neuron, GetLayerIndex(neurons.back()))] << endl;
		//			file << endl;
		//		}

		//		file << endl;
		//	}

		//	cout << endl;
		//	cout << endl;
		//	cout << averageError << endl;
		//	cout << endl;
		//	cout << endl;

		//	stream.close();
		//}

		//string testText;
		//ifstream testStream(argv[1]);
		//int runCount = 0;

		//while (getline(testStream, testText))
		//{
		//	if (runCount == 0)
		//	{
		//		runCount++;
		//		continue;
		//	}

		//	string path;
		//	char expResult = '0';

		//	for (int x = 0; x < testText.length(); x++)
		//	{
		//		if (testText[x] == ',')
		//		{
		//			expResult = testText[x + 2];
		//			break;
		//		}
		//		else
		//		{
		//			if (testText[x] != '\0')
		//			{
		//				path += testText[x];
		//			}
		//		}
		//	}

		//	GetInputsFromImage(argv[2] + path);

		//	FeedForward();

		//	std::string logPath("D:\\Smartlearn\\5_daten\\4. Lehrjahr\\1. Semester\\ABU\\VA\\NeuralNet\\Results\\TestingResults.txt");

		//	fstream file(logPath, ios_base::app | ios_base::out | ios_base::in);

		//	file << bitset<8>(expResult).to_string() << "|" << expResult << endl;

		//	for (Neuron* neuron : neurons.back())
		//	{
		//		file << "Output Value: " << neuron->GetValue() << endl;
		//		file << "Expected Value: " << expOutputs[GetNeuronIndex(neuron, GetLayerIndex(neurons.back()))] << endl;
		//		file << endl;
		//		file << endl;
		//		file << endl;
		//	}
		//}

		//testStream.close();
	}
}