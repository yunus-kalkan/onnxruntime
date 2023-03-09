/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtTrainingSession.OrtCheckpointState;
import ai.onnxruntime.TensorInfo.OnnxTensorType;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Tests for the ORT training apis. */
@EnabledIfSystemProperty(named = "ENABLE_TRAINING", matches = "1")
public class TrainingTest {

  private static final OrtEnvironment env = OrtEnvironment.getEnvironment();

  @Test
  public void testLoadCheckpoint() throws OrtException {
    Path ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt");
    try (OrtCheckpointState ckpt = OrtCheckpointState.loadCheckpoint(ckptPath)) {
      // Must be non-null, exists so the try block isn't empty as this call will
      // throw if it fails, and throwing errors the test
      Assertions.assertNotNull(ckpt);
    }
  }

  @Test
  public void testCreateTrainingSession() throws OrtException {
    String ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(ckptPath, trainPath, null, null)) {
      Assertions.assertNotNull(trainingSession);
      Set<String> inputNames = trainingSession.getTrainInputNames();
      Assertions.assertFalse(inputNames.isEmpty());
      Set<String> outputNames = trainingSession.getTrainOutputNames();
      Assertions.assertFalse(outputNames.isEmpty());
    }
  }

  // this test is not enabled as ORT Java doesn't support supplying an output buffer
  @Disabled
  @Test
  public void TestTrainingSessionTrainStep() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    float[] expectedOutput =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_1.out"));
    float[] input = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/input-0.in"));
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, null)) {
      int[] labels = {1, 1};

      // Run train step with pinned inputs and pinned outputs
      Map<String, OnnxTensor> pinnedInputs = new HashMap<>();
      Map<String, OnnxTensor> outputMap = new HashMap<>();
      try {
        // Create inputs
        long[] inputShape = {2, 784};
        pinnedInputs.put(
            "input-0", OnnxTensor.createTensor(env, OrtUtil.reshape(input, inputShape)));

        // long[] labelsShape = {2};
        pinnedInputs.put("labels", OnnxTensor.createTensor(env, labels));

        // Prepare output buffer
        FloatBuffer output =
            ByteBuffer.allocateDirect(4 * expectedOutput.length)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        OnnxTensor outputTensor =
            OnnxTensor.createTensor(env, output, new long[expectedOutput.length]);
        outputMap.put("onnx::loss::21273", outputTensor);
        /* Disabled as we haven't implemented this yet
        try (trainingSession.trainStep(pinnedInputs, outputMap)) {
          Assertions.assertArrayEquals(expectedOutput, (float[]) outputTensor.getValue(), 1e-3f);
        }
        */
      } finally {
        OnnxValue.close(outputMap);
        OnnxValue.close(pinnedInputs);
      }
    }
  }

  @Test
  public void TestTrainingSessionOptimizerStep() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();
    float[] expectedOutput_1 =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_1.out"));
    float[] expectedOutput_2 =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_2.out"));
    float[] input = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/input-0.in"));
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
      int[] labels = {1, 1};

      // Run train step with pinned inputs and pinned outputs
      Map<String, OnnxTensor> pinnedInputs = new HashMap<>();
      try {
        // Create inputs
        long[] inputShape = {2, 784};
        pinnedInputs.put(
            "input-0", OnnxTensor.createTensor(env, OrtUtil.reshape(input, inputShape)));

        // long[] labelsShape = {2};
        pinnedInputs.put("labels", OnnxTensor.createTensor(env, labels));

        try (OrtSession.Result outputs = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertEquals(expectedOutput_1[0], (float) outputs.get(0).getValue(), 1e-3f);
        }

        trainingSession.lazyResetGrad();

        try (OrtSession.Result outputs = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertEquals(expectedOutput_1[0], (float) outputs.get(0).getValue(), 1e-3f);
        }

        trainingSession.optimizerStep();

        try (OrtSession.Result outputs = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertEquals(expectedOutput_2[0], (float) outputs.get(0).getValue(), 1e-3f);
        }
      } finally {
        OnnxValue.close(pinnedInputs);
      }
    }
  }

  @Test
  public void TestTrainingSessionSetLearningRate() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();

    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
      float learningRate = 0.245f;
      trainingSession.setLearningRate(learningRate);
      float actualLearningRate = trainingSession.getLearningRate();
      Assertions.assertEquals(learningRate, actualLearningRate);
    }
  }
}
