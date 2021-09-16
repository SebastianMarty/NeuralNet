#include "Neuron.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{
	vector<double> inputs{ 1.0f, 0.3f, 0.4f, 1.0f, 0.5f };
	vector<Neuron*> neurons;

	for (double input : inputs)
	{
		Neuron neuron(input, NeuronType::INPUT);
	}
}