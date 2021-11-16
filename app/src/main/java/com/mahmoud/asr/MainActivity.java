package com.mahmoud.asr;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.jlibrosa.audio.JLibrosa;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;
    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;
    private long lastProcessingTimeMs;
    private TextView inferencetimeinfo;
    private TextView inferencetimeperword;
    private TextView numberofwords;
    private TextView inferencetimepercharacter;
    protected ListView deviceView;
    protected int defaultDeviceIndex = 0;
    String[] ACC_List;

    /** Current indices of device and model. */
    int currentDevice = -1;
    int currentModel = -1;
    int currentNumThreads = -1;

    private String wavFilename;
    private MediaPlayer mediaPlayer = new MediaPlayer();

    private final static String TAG = "TfliteASRDemo";
    private final static int SAMPLE_RATE = 16000 ;
    private final static int DEFAULT_AUDIO_DURATION = -1 ;

    private final static String[] WAV_FILENAMES = {"audio_clip_1.wav", "audio_clip_2.wav", "audio_clip_3.wav"};
    private final static String TFLITE_FILE = "CONFORMER.tflite";
    ArrayList<String> deviceStrings = new ArrayList<String>();
    protected ArrayList<String> modelStrings = new ArrayList<String>();

    private LinearLayout bottomSheetLayout;
    protected ImageView bottomSheetArrowImageView;
    private LinearLayout gestureLayout;
    private BottomSheetBehavior<LinearLayout> sheetBehavior;

    protected int defaultModelIndex = 0;

    protected ListView modelView;
    private static final String ASSET_PATH = "";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        JLibrosa jLibrosa = new JLibrosa();
        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        ArrayAdapter<String>adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);

        playAudioButton = findViewById(R.id.play);
        playAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try(AssetFileDescriptor assetFileDescriptor = getAssets().openFd(wavFilename)) {
                    mediaPlayer.reset();
                    mediaPlayer.setDataSource(assetFileDescriptor.getFileDescriptor(), assetFileDescriptor.getStartOffset(), assetFileDescriptor.getLength());
                    mediaPlayer.prepare();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                mediaPlayer.start();
            }
        });
        transcribeButton = findViewById(R.id.recognize);
        resultTextview = findViewById(R.id.result);
        inferencetimeinfo = findViewById(R.id.inference_time_info);
        inferencetimeperword = findViewById(R.id.inf_per_word_info);
        numberofwords = findViewById(R.id.number_word_info);
        inferencetimepercharacter = findViewById(R.id.number_char_info);

        // SET ACC (Device) LIST
        deviceView = findViewById(R.id.device_list);
        deviceStrings.add("CPU");
        deviceStrings.add("GPU");
        deviceStrings.add("NNAPI");
        deviceView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        ArrayAdapter<String> deviceAdapter =
                new ArrayAdapter<>(
                        this , R.layout.deviceview_row, R.id.deviceview_row_text, deviceStrings);
        deviceView.setAdapter(deviceAdapter);
        deviceView.setItemChecked(defaultDeviceIndex, true);
        currentDevice = defaultDeviceIndex;
        deviceView.setOnItemClickListener(
                new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                       // updateActiveModel();
                    }
                });

        //List of Models[conformer, tranducer, contextnet]
        bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
        gestureLayout = findViewById(R.id.gesture_layout);
        sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
        // bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
        modelView = findViewById((R.id.model_list));

        modelStrings = getModelStrings(getAssets(), ASSET_PATH);
        modelView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        ArrayAdapter<String> modelAdapter =
                new ArrayAdapter<>(
                        this , R.layout.listview_row, R.id.listview_row_text, modelStrings);
        modelView.setAdapter(modelAdapter);
        modelView.setItemChecked(defaultModelIndex, true);
        currentModel = defaultModelIndex;
        modelView.setOnItemClickListener(
                new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                        // updateActiveModel();
                    }
                });


        // TRANSCRIBTION
        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View v) {
                try {
                    float audioFeatureValues[] = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);
                    Object[] inputArray = {audioFeatureValues};
                    IntBuffer outputBuffer = IntBuffer.allocate(2000);

                    Map<Integer, Object> outputMap = new HashMap<>();
                    outputMap.put(0, outputBuffer);

                    tfLiteModel = loadModelFile(getAssets(), TFLITE_FILE);
                    long startTime = new Date().getTime();
                    Interpreter.Options tfLiteOptions = new Interpreter.Options();
                    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
                    tfLite.resizeInput(0, new int[] {audioFeatureValues.length});
                    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);


                    int outputSize = tfLite.getOutputTensor(0).shape()[0];
                    int[] outputArray = new int[outputSize];
                    outputBuffer.rewind();
                    outputBuffer.get(outputArray);
                    StringBuilder finalResult = new StringBuilder();
                    for(int i=0; i < outputSize; i++){
                        char c = (char) outputArray[i];
                        if (outputArray[i] != 0){
                            finalResult.append((char) outputArray[i]);
                        }
                    }
                    lastProcessingTimeMs = new Date().getTime() - startTime;
                    String finalResult_str = finalResult.toString();
                    Log.v("output", "result"+ finalResult_str);
                    String[] splited = finalResult_str.split("\\s+");
                    // long totalCharacters = finalResult_str.chars().filter(ch -> ch != ' ').count();
                    long totalCharacters = finalResult_str.chars().count();
                    int num_of_words = splited.length;
                    Log.v("output", "now"+ num_of_words);
                    int inf_per_word = (int) (lastProcessingTimeMs / num_of_words);
                    int inf_per_char = (int) (lastProcessingTimeMs / totalCharacters);
                    Log.v("output", "infperchar"+ inf_per_char);

                    resultTextview.setText(finalResult.toString());
                    numberofwords.setText(num_of_words + "  ");
                    inferencetimeinfo.setText(lastProcessingTimeMs + "ms");
                    inferencetimeperword.setText(inf_per_word + "ms");
                    inferencetimepercharacter.setText(inf_per_char + "ms");
                } catch (Exception e){
                    Log.e(TAG, e.getMessage());
                }

            }
        });
    }

    // List of models
    protected ArrayList<String> getModelStrings(AssetManager mgr, String path){
        ArrayList<String> res = new ArrayList<String>();
        try {
            String[] files = mgr.list(path);
            for (String file : files) {
                String[] splits = file.split("\\.");
                if (splits[splits.length - 1].equals("tflite")) {
                    res.add(file);
                }
            }

        }
        catch (IOException e){
            System.err.println("getModelStrings: " + e.getMessage());
        }
        return res;
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
        throws IOException{
            AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

        private String copyWavFileToCache(String wavFilename) {
            File destinationFile = new File(getCacheDir() + wavFilename);
            if (!destinationFile.exists()) {
                try {
                    InputStream inputStream = getAssets().open(wavFilename);
                    int inputStreamSize = inputStream.available();
                    byte[] buffer = new byte[inputStreamSize];
                    inputStream.read(buffer);
                    inputStream.close();

                    FileOutputStream fileOutputStream = new FileOutputStream(destinationFile);
                    fileOutputStream.write(buffer);
                    fileOutputStream.close();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }

            return getCacheDir() + wavFilename;
        }

    }

