// src/utils/preprocess.ts
import { Buffer } from 'buffer';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';

/**
 * Resize image to 640x640
 * Convert image to Float32 tensor
 * Normalize values to 0-1
 */

const INPUT_SIZE = 640;

export async function preprocessImage(
  imageUri: string,
): Promise<Float32Array> {
  try {
    /**
     * Resize image
     */
    const resizedImage =
      await ImageResizer.createResizedImage(
        imageUri,
        INPUT_SIZE,
        INPUT_SIZE,
        'JPEG',
        100,
      );

    /**
     * Read resized image as base64
     */
    const base64 =
      await RNFS.readFile(
        resizedImage.uri,
        'base64',
      );

    /**
     * Convert base64 -> buffer
     */
    const imageBuffer = Buffer.from(
      base64,
      'base64',
    );

    /**
     * VERY IMPORTANT:
     *
     * This is NOT true RGB decoding.
     *
     * For production YOLO:
     * use react-native-vision-camera
     * OR react-native-opencv
     *
     * This is simplified tensor creation
     * for testing model inference.
     */

    const floatArray = new Float32Array(
      1 * 3 * INPUT_SIZE * INPUT_SIZE,
    );

    /**
     * Fake RGB extraction
     * (placeholder preprocessing)
     */
    for (
      let i = 0;
      i < floatArray.length;
      i++
    ) {
      const value =
        imageBuffer[
          i % imageBuffer.length
        ];

      floatArray[i] = value / 255.0;
    }

    return floatArray;
  } catch (error) {
    console.error(
      'Preprocess error:',
      error,
    );

    throw error;
  }
}