get_dataset:
	wget https://dl.fbaipublicfiles.com/glue/data/CoLA.zip
	mv CoLA.zip ./data/CoLA.zip
	unzip ./data/CoLA.zip -d ./data/
	rm ./data/CoLA.zip
