use rand::{seq::SliceRandom, thread_rng};
use serde::Deserialize;
use std::{convert::TryInto, fs::File, path::Path};
use tensorflow::{
    ops, train, train::Optimizer, DataType, Scope, Session, SessionOptions, SessionRunArgs, Shape,
    Tensor, Variable,
};

type BoxedError = Box<dyn std::error::Error>;

/// A struct to hold our MNIST dataset
#[derive(Deserialize, Default)]
struct MnistDigit {
    pub label: Vec<f32>,
    pub pixels: Vec<f32>,
}
// Path to MNIST training data (CSV)
const TRAIN_DATA_PATH: &'static str = "path/to/mnist_train.csv";
// Path to MNIST test data (CSV)
const TEST_DATA_PATH: &'static str = "path/to/mnist_test.csv";

// Number of input nodes to the NN, one for each pixel (28*28 = 784)
const INPUT_SIZE: i64 = 784;
// Number of nodes in the hidden layers
const HIDDEN_LAYER_SIZE: i64 = 500;
// The NN will have 10 output nodes, one for each digit 0-9
const OUTPUT_SIZE: i64 = 10;
// This is the learning rate
const LEARNING_RATE: f32 = 0.0001;
// We split the total number of images in batches
// of this size during training.
const BATCH_SIZE: i64 = 100;
// Number of times we run through the training
const EPOCHS: i64 = 75;

fn main() -> Result<(), BoxedError> {
    // Load the MNIST dataset
    let mnist_train = load_dataset(Path::new(TRAIN_DATA_PATH))?;
    let mnist_test = load_dataset(Path::new(TEST_DATA_PATH))?;

    // Build the model.
    // A scope groups operations together and is
    // supplied as a parameter when creating operations (ops).
    // This is the root scope of the entire graph.
    let mut root_scope = Scope::new_root_scope();
    let scope = &mut root_scope;

    // Placeholder for input layer.
    // Placeholders are just that; a placeholder.
    // We prepare them as part of the network now,
    // but don't send the actual values unless we train
    // or evaluate the network.
    //
    // When we build our network operations, we can optionally name
    // them with the scope variable. The name shows up in the TensorFlow
    // output and can be useful for debugging. I will name each operation
    // the same as the Rust variable for this guide.
    let network_input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape(Shape::from(Some(vec![None, Some(INPUT_SIZE)])))
        .build(&mut scope.with_op_name("network_input"))?;

    // Another placeholder for the output layer.
    let target_output = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape(Shape::from(Some(vec![None, Some(OUTPUT_SIZE)])))
        .build(&mut scope.with_op_name("target_output"))?;
    // First hidden layer.
    // The results from creating the layers represents the weights between the layers.
    // We initialize these with random values from a normal distribution.

    // Although variables are constructed now, and instructed what
    // value they should have, they are not initialized until later.
    // We store all the variables in a collection so we can use them later,
    // for initialization and for our optimizer.
    let mut variables = Vec::new();

    // For the ops in the layers, we use a sup scope.
    let mut layer_scope = scope.new_sub_scope("layer");
    let layer_scope = &mut layer_scope;
    let weights_1_shape = ops::constant(&[INPUT_SIZE, HIDDEN_LAYER_SIZE][..], layer_scope)?;

    let layer_1 = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(weights_1_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[INPUT_SIZE, HIDDEN_LAYER_SIZE][..]))
        .build(&mut layer_scope.with_op_name("layer_1"))?;
    variables.push(layer_1.clone());

    // First layer bias.
    let bias_1_shape = ops::constant(&[HIDDEN_LAYER_SIZE][..], layer_scope)?;
    let layer_1_bias = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(bias_1_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[HIDDEN_LAYER_SIZE][..]))
        .build(&mut layer_scope.with_op_name("layer_1_bias"))?;
    variables.push(layer_1_bias.clone());
    // Second hidden layer
    let weights_2_shape = ops::constant(&[HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE][..], layer_scope)?;

    let layer_2 = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(weights_2_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE][..]))
        .build(&mut layer_scope.with_op_name("layer_2"))?;
    variables.push(layer_2.clone());
    // Second layer bias.
    let bias_2_shape = ops::constant(&[HIDDEN_LAYER_SIZE][..], layer_scope)?;
    let layer_2_bias = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(bias_2_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[HIDDEN_LAYER_SIZE][..]))
        .build(&mut layer_scope.with_op_name("layer_2_bias"))?;
    variables.push(layer_2_bias.clone());
    // Third hidden layer
    let weights_3_shape = ops::constant(&[HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE][..], layer_scope)?;

    let layer_3 = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(weights_3_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE][..]))
        .build(&mut layer_scope.with_op_name("layer_3"))?;
    variables.push(layer_3.clone());
    // Third layer bias.
    let bias_3_shape = ops::constant(&[HIDDEN_LAYER_SIZE][..], layer_scope)?;
    let layer_3_bias = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(bias_3_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[HIDDEN_LAYER_SIZE][..]))
        .build(&mut layer_scope.with_op_name("layer_3_bias"))?;
    variables.push(layer_3_bias.clone());
    // Output layer
    let weights_output_shape = ops::constant(&[HIDDEN_LAYER_SIZE, OUTPUT_SIZE][..], layer_scope)?;

    let out_layer = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(weights_output_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[HIDDEN_LAYER_SIZE, OUTPUT_SIZE][..]))
        .build(&mut layer_scope.with_op_name("out_layer"))?;
    variables.push(out_layer.clone());
    // Output layer bias.
    let out_bias_shape = ops::constant(&[OUTPUT_SIZE][..], layer_scope)?;
    let out_layer_bias = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(out_bias_shape.into(), layer_scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[OUTPUT_SIZE][..]))
        .build(&mut layer_scope.with_op_name("out_layer_bias"))?;
    variables.push(out_layer_bias.clone());
    // All placeholders, shapes and weights are now prepared for the network.
    // Time to add the calculations for moving through our network.
    // We will use a ReLu activation function.
    // The inputs of the ReLu are operations, we can nest these
    // but for readability we split them up and create them one by one

    // Activation for first layer
    let layer_1_mul = ops::mat_mul(
        network_input.clone().into(),
        layer_1.output().clone(),
        layer_scope,
    )?;
    let layer_1_with_bias = ops::add(
        layer_1_mul.into(),
        layer_1_bias.output().clone(),
        layer_scope,
    )?;
    let layer_1_output = ops::relu(
        layer_1_with_bias.into(),
        &mut layer_scope.with_op_name("layer_1_output"),
    )?;

    // Activation for second layer
    let layer_2_mul = ops::mat_mul(layer_1_output.into(), layer_2.output().clone(), layer_scope)?;
    let layer_2_with_bias = ops::add(
        layer_2_mul.into(),
        layer_2_bias.output().clone(),
        layer_scope,
    )?;
    let layer_2_output = ops::relu(
        layer_2_with_bias.into(),
        &mut layer_scope.with_op_name("layer_2_output"),
    )?;

    // Activation for third layer
    let layer_3_mul = ops::mat_mul(layer_2_output.into(), layer_3.output().clone(), layer_scope)?;
    let layer_3_with_bias = ops::add(
        layer_3_mul.into(),
        layer_3_bias.output().clone(),
        layer_scope,
    )?;
    let layer_3_output = ops::relu(
        layer_3_with_bias.into(),
        &mut layer_scope.with_op_name("layer_3_output"),
    )?;

    // First network output - raw logits
    // Raw logits are later fed to softmax
    let network_output_mul = ops::mat_mul(
        layer_3_output.into(),
        out_layer.output().clone(),
        &mut layer_scope.with_op_name("network_output_mul"),
    )?;
    let network_output_1 = ops::add(
        network_output_mul.clone().into(),
        out_layer_bias.output().clone(),
        &mut layer_scope.with_op_name("network_output_1"),
    )?;

    // Second network output - softmax
    let network_output_2 = ops::softmax(
        network_output_1.clone().into(),
        &mut layer_scope.with_op_name("network_output_2"),
    )?;

    // Now it's time to prepare the training.
    // Let's add a new sub scope for this
    let mut training_scope = scope.new_sub_scope("training");
    let training_scope = &mut training_scope;
    let softmax_cewl = ops::softmax_cross_entropy_with_logits(
        network_output_1.clone().into(),
        target_output.clone().into(),
        &mut training_scope.with_op_name("softmax_cewl"),
    )?;
    // Some operations require a dimension parameter, we will mostly use
    // 0 for this so we prepare a constant and re-use it where required.
    let const_0 = ops::constant(0, &mut training_scope.with_op_name("const_0"))?;
    // Our cost function
    let cost_function = ops::mean(
        softmax_cewl.into(),
        const_0.clone().into(),
        &mut training_scope.with_op_name("cost_function"),
    )?;
    // Create a constant operation to hold our learning rate
    let lr = ops::constant(LEARNING_RATE, training_scope)?;
    // This is our optimizer. We feed it our learning rate and also
    // all our defined variables.
    let training_step = train::GradientDescentOptimizer::new(lr).minimize(
        &mut training_scope.with_op_name("training_step"),
        cost_function.clone().into(),
        train::MinimizeOptions::default().with_variables(&variables),
    )?;
    variables.extend(training_step.0);
    // The following section uses arg_max to find the correct index
    // in our output array. Remember our output is an array with ten floats:
    // [0.0, 0.0, .... 0.0]
    // By using argmax with 1 for the output and target labels, we can get
    // the index of the number.
    let const_1 = ops::constant(1, &mut training_scope.with_op_name("const_0"))?;
    let argmax_network_output_2 = ops::arg_max(
        network_output_2.clone().into(),
        const_1.clone().into(),
        &mut training_scope.with_op_name("argmax_network_output_2"),
    )?;
    let argmax_target_output = ops::arg_max(
        target_output.clone().into(),
        const_1.clone().into(),
        &mut training_scope.with_op_name("argmax_target_output"),
    )?;
    let correct_predictions = ops::equal(
        argmax_network_output_2.into(),
        argmax_target_output.into(),
        &mut training_scope.with_op_name("correct_predictions"),
    )?;
    // We cast our correct predictions to a float and use it for accuracy
    let cast = ops::Cast::new()
        .SrcT(DataType::Bool)
        .DstT(DataType::Float)
        .build(correct_predictions.clone().into(), training_scope)?;
    let accuracy = ops::mean(
        cast.into(),
        const_0.clone().into(),
        &mut training_scope.with_op_name("accuracy"),
    )?;

    // Time to run the graph.
    // Whenever we want to actually run our network/graph, we do
    // so using a session.
    let session = Session::new(&SessionOptions::new(), &scope.graph_mut())?;
    // We re-use the same session, but can control it using SessionRunArgs.
    // Here, we tell it to initialize all variables.
    let mut run_args = SessionRunArgs::new();

    // Initialize all variables
    for var in &variables {
        run_args.add_target(&var.initializer());
    }
    session.run(&mut run_args)?;
    // Ok, all variables are initialized. Let's start training.
    // Our outer loop (EPOCHS) is the number of times we will run
    // through our entire training set.
    for epoch in 0..EPOCHS {
        let mut total_cost = 0.0;
        let mnist_iter = mnist_train.iter();
        // For each EPOCH, we split the training set in BATCH_SIZE batches.
        for i in 0..(mnist_train.len() as i64 / BATCH_SIZE) {
            let batch: Vec<&MnistDigit> = mnist_iter
                .clone()
                .skip((i * BATCH_SIZE).try_into().unwrap())
                .take(BATCH_SIZE as usize)
                .collect();
            // batch_x is our features (pixel data)
            let batch_x: Vec<Vec<f32>> = batch.iter().map(|b| b.pixels.clone()).collect();
            // batch_y is our labels
            let batch_y: Vec<Vec<f32>> = batch.iter().map(|b| b.label.clone()).collect();
            // Remember the placeholders we defined way up in the beginning?
            // Now it's time to feed them with data.
            //
            // The pixel values (our network input) are fed into the network
            // with a tensor of floats, with shape [BATCH_SIZE, 784].
            let input_tensor =
                Tensor::<f32>::new(&[BATCH_SIZE as u64, 784]).with_values(&batch_x.concat())?;
            // The labels are also a tensor floats, but with the shape of
            // [BATCH SIZE, 10].
            let label_tensor =
                Tensor::<f32>::new(&[BATCH_SIZE as u64, 10]).with_values(&batch_y.concat())?;
            // Now we specify the arguments for running the network. We want to be able
            // to get the cost/loss, so we can see that it is decreasing.
            // The way to get a specific operation from the graph is by using request_fetch
            let mut run_args = SessionRunArgs::new();
            // Here we get a token for the output of the cost function.
            // This token can be used after we run the network to get the actual value.
            let cf_fetch = run_args.request_fetch(&cost_function.clone(), 0);
            // The target for training is our training step operation.
            run_args.add_target(&training_step.1);
            // We add our input and output as feeds.
            run_args.add_feed(&network_input, 0, &input_tensor);
            run_args.add_feed(&target_output, 0, &label_tensor);
            // And run the graph.
            session.run(&mut run_args)?;
            // Here we use our previously defined token to get the actual
            // output from, in this case, the cost function.
            // For debugging and learning you can set up more of these
            // and inspect the output of various operations.
            let cost = run_args.fetch::<f32>(cf_fetch)?[0];
            // We sum the total cost/loss for every EPOCH
            total_cost += cost;
        }
        // We print out a status message as we're moving through
        // the epochs. The idea here is to see that the loss is
        // decreasing, as expected when using gradient descent.
        println!(
            "Epoch {} completed out of {}, loss: {:.2}",
            epoch + 1,
            EPOCHS,
            total_cost,
        );
    }
    // OK, we are done training! Let's check the accuracy.
    // This time we use the entire test set of MNIST,
    // mapping it to input and output of the network just like when training.
    // Same shapes, different values.
    let test_x: Vec<Vec<f32>> = mnist_test.iter().map(|b| b.pixels.clone()).collect();
    let test_y: Vec<Vec<f32>> = mnist_test.iter().map(|b| b.label.clone()).collect();
    let input_tensor =
        Tensor::<f32>::new(&[mnist_test.len() as u64, 784]).with_values(&test_x.concat())?;
    let label_tensor =
        Tensor::<f32>::new(&[mnist_test.len() as u64, 10]).with_values(&test_y.concat())?;
    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(&network_input, 0, &input_tensor);
    run_args.add_feed(&target_output, 0, &label_tensor);
    run_args.add_target(&accuracy);
    // We're really only interested in the accuracy now,
    // add a fetch token for it.
    let acc_fetch = run_args.request_fetch(&accuracy, 0);
    // Run the graph.
    session.run(&mut run_args)?;
    // Get the resulting accuracy, and print it
    let acc = run_args.fetch::<f32>(acc_fetch)?[0];
    println!("Accuracy: {:.1}%", acc * 100.0);
    Ok(())
}

/// This loads CSV MNIST data from the supplied path.
/// The input is expected to be 784 pixel values (0-255) followed
/// by the actual number (0-9)
///
/// We translate the pixel values from 0-255 to 0.0-1.0
/// and the actual number from a single digit to an array
/// filled with zeroes, except in the index for the number.
///
/// So a label with number 3 would be:
/// [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fn load_dataset(p: &Path) -> Result<Vec<MnistDigit>, BoxedError> {
    let f = File::open(p)?;
    let mut reader = csv::Reader::from_reader(f);

    let mut digits: Vec<MnistDigit> = reader
        .records()
        .map(|r| {
            let record = r.unwrap();
            let mut digit = MnistDigit::default();
            let number = record[0].parse::<usize>().unwrap();
            let mut label_array = vec![0.0; 10];
            label_array[number] = 1.0;
            digit.label = label_array;
            digit.pixels = record
                .iter()
                .skip(1)
                .map(|ir| ir.parse::<f32>().unwrap() / 255.0)
                .collect();
            digit
        })
        .collect();
    // Shuffle
    digits.shuffle(&mut thread_rng());

    Ok(digits)
}
