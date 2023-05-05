# Adams
![adams](https://user-images.githubusercontent.com/108131461/236461064-64b79968-890c-4ae0-a7b9-d950ec669bf5.png)

Adams is a personal language model based on the GPT-2 architecture that has been fine-tuned on a dataset of web pages scraped from Google search results. It is designed to be used as a chatbot or conversational agent that can generate human-like responses to user input.

## Release Notes
This release of Adams includes several new features and improvements, including:

- Improved training data preprocessing for better text cleaning and filtering
- Integration with a Google search API to obtain search results more efficiently
- Optimized training hyperparameters to reduce training time and improve model performance
- Enhanced error handling and logging capabilities

## Installation
To install Adams, simply clone this repository to your local machine:

```bash
git clone https://github.com/somkietacode/adams.git
```
Then, navigate to the project directory and install the necessary dependencies using pip:

```bash
cd adams
pip install -r requirements.txt
```

## Usage
To use Adams as a chatbot or conversational agent, simply run the adams.py script:

```bash
python adams.py
```

This will start a command-line interface where you can enter text input and receive responses generated by Adams.

## Demo
```bash
User > GPT-2
{'loss': 1.8762, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 118.081, 'train_samples_per_second': 0.076, 'train_steps_per_second': 0.008, 'train_loss': 1.8762047290802002, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [01:57<00:00, 117.52s/it]
Model saved as adam.model
User > Tell me a duck story
{'loss': 1.5394, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 96.461, 'train_samples_per_second': 0.093, 'train_steps_per_second': 0.01, 'train_loss': 1.5393702983856201, 'epoch': 1.0}
100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:36<00:00, 96.30s/it]
Model saved as adam.model
Bot > Tell me a duck story.
Bot > A duck is a creature that has a shell on it.
Bot > It has the ability to move at high speeds, and can move up and down at will.
Bot > This makes it a good choice for those who want to use it for a variety of other tasks, such as hunting, or gathering food.
User > About yesterday
{'loss': 1.2021, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 113.638, 'train_samples_per_second': 0.079, 'train_steps_per_second': 0.009, 'train_loss': 1.202134609222412, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [01:53<00:00, 113.28s/it]
Model saved as adam.model
Bot > About yesterday, I wrote a blog post about how to make a video game.
Bot > Today, we're going to talk about making a movie.
Bot > Let's start with the basics.
Bot > First, let's take a look at the video.
Bot > The video starts with a simple premise: "I want to watch this video."
Bot > This is the premise that video is about.
User > What is a bot?
{'loss': 0.923, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 142.424, 'train_samples_per_second': 0.063, 'train_steps_per_second': 0.007, 'train_loss': 0.923012375831604, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [02:22<00:00, 142.23s/it]
Model saved as adam.model
Bot > What is a bot?
Bot > A bot is an automated process where a user creates a new bot using a simple command-and-line interface.
Bot > The bot can then be used for other tasks such as text processing, image recognition, and so on.
Bot > Bot-based Model (BBM) models are built on top-notch Python code, making them easy to use and maintain.
User > What is even?
{'loss': 0.7361, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 145.5799, 'train_samples_per_second': 0.062, 'train_steps_per_second': 0.007, 'train_loss': 0.736084520816803, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [02:25<00:00, 145.12s/it]
Model saved as adam.model
Bot > What is even?
Bot > It's a simple question: What is a "real-world" game?
Bot > The answer is simple: It's own set of tools, which can be applied to any kind of game.
User > Define a dog
{'loss': 0.478, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 63.512, 'train_samples_per_second': 0.142, 'train_steps_per_second': 0.016, 'train_loss': 0.47801339626312256, 'epoch': 1.0}
100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.12s/it]
Model saved as adam.model
Bot > Define a dog as a member of the same-sex couples.
Bot > In addition to the above examples, we've also included a few other examples of relationships between dogs and humans.
Bot > These relationships can range from simple to complex relationships such as friendship, family relationships, or even relationships with other animals.
User > What is Paris?
{'loss': 0.4311, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 70.3227, 'train_samples_per_second': 0.128, 'train_steps_per_second': 0.014, 'train_loss': 0.43113067746162415, 'epoch': 1.0}
100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:09<00:00, 69.89s/it]
Model saved as adam.model
Bot > What is Paris?
Bot > Paris is a city, with a wide variety of attractions, including museums, galleries, and more.
Bot > It has become one of the most popular tourist destinations in the world, thanks to its wide range of different types of visitors and the fact that it is open year-round.
User >
```



## Contributing
If you would like to contribute to Adams, please follow these steps:

- Fork the repository and create a new branch for your feature or bug fix
- Make your changes and write tests to cover any new functionality
- Run the tests and ensure that they all pass
- Submit a pull request describing your changes and why they should be merged into the main branch

## License
Adams is released under the MIT License.
