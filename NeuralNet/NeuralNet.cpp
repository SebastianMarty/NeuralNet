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
#include <bitset>
#include <Windows.h>

using namespace std;

class Neuron;
typedef std::vector<Neuron> Layer;

int trainingRounds = 40;
vector<double> inputs;
vector<double> expOutputs;
class Network* net;
vector<unsigned> layout{ 1024, 512, 8 };

void GetInputsFromImage(string path)
{
	cv::Mat image = cv::imread(path);

	if (!image.empty())
	{
		//Resize image to fixed input size
		cv::Mat imgResized;
		cv::resize(image, imgResized, cv::Size(32, 32), cv::InterpolationFlags::INTER_LINEAR);

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
	void SetWeights(std::vector<Connection> weights) {outputWeights = weights}
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

char ShowOptions()
{
	std::cout << "1. Create new Network" << std::endl;
	std::cout << "2. Load Network" << std::endl;
	std::cout << "3. Recognize image" << std::endl;
	std::cout << "4. Save Network" << std::endl;
	std::cout << "5. Save and Quit" << std::endl;

	std::string choice;
	std::getline(std::cin, choice);

	return choice[0];
}

void ClearConsole()
{
	COORD topLeft = { 0, 0 };
	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO screen;
	DWORD written;

	GetConsoleScreenBufferInfo(console, &screen);
	FillConsoleOutputCharacterA(
		console, ' ', screen.dwSize.X * screen.dwSize.Y, topLeft, &written
	);
	FillConsoleOutputAttribute(
		console, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE,
		screen.dwSize.X * screen.dwSize.Y, topLeft, &written
	);
	SetConsoleCursorPosition(console, topLeft);
}

void SaveNetwork()
{
	std::string path;

	std::cout << "Where should the network be saved? ";
	std::getline(std::cin, path);
	std::cout << std::endl;

	ofstream file(path);

	vector<Layer> layers = net->getLayers();

	file << std::endl;

	for (unsigned x = 0; x < layout.size(); x++)
	{
		file << layout[x];

		if (x + 1 != layout.size())
		{
			file << " ";
		}
		else
		{
			file << std::endl;
		}
	}

	for (unsigned x = 0; x < layers.size(); x++)
	{
		for (unsigned y = 0; y < layers[x].size(); y++)
		{
			vector<Connection> outputWeights = layers[x][y].getOutputWeights();

			for (unsigned z = 0; z < outputWeights.size(); z++)
			{
				file << outputWeights[z].weight;

				if (z + 1 != outputWeights.size())
				{
					file << " ";
				}
				else
				{
					file << std::endl;
				}
			}
		}
	}

	file.close();
}

Network* LoadNetwork()
{
	std::string path;

	std::cout << "Enter the path to a saved network: ";
	std::getline(std::cin, path);
	std::cout << std::endl;

	std::string text;
	ifstream file(path);
	int runCount = 0;
	std::string value;
	std::vector<Connection> weights;

	layout = {};

	while (std::getline(file, text))
	{
		if (runCount == 0)
		{
			runCount++;
			continue;
		}

		if (runCount == 1)
		{
			for (int x = 0; x < text.length(); x++)
			{
				if (text[x] != ' ')
				{
					value += text[x];
				}
				else
				{
					layout.push_back(atoi(value.c_str()));
				}
			}
		}

		for (int x = 0; x < text.length(); x++)
		{
			if (text[x] != ' ')
			{
				value += text[x];
			}
			else
			{
				weights.push_back(Connection().weight = atof(value.c_str()));
			}
		}

		runCount++;
	}

	return new Network(layout);
}

int main(int argc, char* argv[])
{
	while (true)
	{
		std::string choice = "0";

		if (choice[0] == '0')
		{
			choice = ShowOptions();
		}

		switch (choice[0])
		{
		case '1':
		{
			ClearConsole();

			std::string trainingResultsPath;
			std::string trainingImagesPath;

			std::cout << "Enter path to training data file: ";
			std::getline(std::cin, trainingResultsPath);
			std::cout << std::endl;

			std::cout << "Enter path to training data image folder: ";
			std::getline(std::cin, trainingImagesPath);
			std::cout << std::endl;
			ClearConsole();

			srand(time(NULL));

			net = new Network(layout);

			for (int x = 0; x < trainingRounds; x++)
			{
				string text;
				ifstream stream(trainingResultsPath);
				int runCount = 0;

				while (std::getline(stream, text))
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

					vector<double> expValues;

					for (int c : bitset<8>(expResult).to_string())
					{
						expValues.push_back(c - 48);
					}

					expOutputs = expValues;
					expValues = {};

					std::cout << "Iteration: " << x + 1 << std::endl;

					GetInputsFromImage(trainingImagesPath + path);
					net->feedForward(inputs);

					net->backProp(expOutputs);

					std::cout << std::endl << "Targets: " << std::flush;

					for (unsigned i = 0; i < expOutputs.size(); i++)
						std::cout << expOutputs[i] << " " << std::flush;

					std::cout << std::endl << "Results: " << std::flush;

					std::vector<double> results;
					net->getResults(results);

					for (unsigned i = 0; i < results.size(); i++)
						std::cout << results[i] << " " << std::flush;

					std::cout << std::endl << "Average recent error: " << net->getRecentAverageError() << std::endl << std::endl;

					if (x + 1 == trainingRounds) {
						std::string input;
						std::cout << "Neural network has reached expected amount of iterations." << std::endl;
						std::cout << "Enter Y to save the neural network, enter N to keep training it" << std::endl;
						std::getline(std::cin, input);
						std::cin.ignore(256, '\n');
						if (input == "N" || input == "n")
						{
							x = 0;
						}
						else if (input == "Y" || input == "y")
						{
							break;
						}
					}
				}
			}
			break;
		}
		case '2':
		{
			std::string netPath;

			std::cout << "Enter path to safed Network file: ";
			std::getline(std::cin, netPath);
			std::cout << std::endl;

			net = LoadNetwork();

			srand(time(NULL));

			net = new Network(layout);

			break;
		}
		case '3':
		{
			do
			{
				std::string imagePath;
				std::cout << "1. Recognize Image" << std::endl;
				std::cout << "2. Back" << std::endl;
				std::getline(std::cin, choice);
				ClearConsole();

				if (choice[0] != '1' && choice[0] != '2')
				{
					std::cout << "Invalid input" << std::endl;
					continue;
				}

				if (choice[0] == '2')
				{
					break;
				}

				std::cout << "Enter path to image that should be recognized: ";
				std::getline(std::cin, imagePath);
				std::cout << std::endl;

				GetInputsFromImage(imagePath);
				net->feedForward(inputs);

				std::vector<double> results;
				net->getResults(results);

				std::cout << "Result: ";

				for (unsigned i = 0; i < results.size(); i++)
					std::cout << results[i] << " " << std::flush;

				std::cout << std::endl;
			} while (choice[0] != 2);

			break;
		}
		case '4':
			SaveNetwork();
			break;
		case '5':
			SaveNetwork();
			return 0;
			break;
		default:
			ClearConsole();
			std::cout << "Invalid Input" << std::endl;
			break;
		}

		choice = '0';
	}
}