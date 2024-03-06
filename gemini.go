package main

import (
	"context"
	"fmt"
	"log"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

const ImageTemperture = 0.8
const ChatTemperture = 0.3

// Gemini Image: Input an image and get the response string.
func GeminiImage(imgData []byte) (string, error) {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-pro-vision")
	value := float32(ImageTemperture)
	model.Temperature = &value
	prompt := []genai.Part{
		genai.ImageData("png", imgData),
		genai.Text("As an expert specializing in diabetic diatician and assessing the suitability of fruits and foods for individuals with diabetes, your task involves analyzing input images featuring various food items. Your first objective is to identify the type of fruit or food present in the image. Subsequently, you must determine the glycemic index and the estimation of calories of the identified item. Based on this glycemic index and calories, provide recommendations on whether individuals with diabetes can include the detected food in their diet. If the food is deemed suitable, specify the recommended quantity for consumption. response all in zh-tw"),
	}
	log.Println("Begin processing image...")
	resp, err := model.GenerateContent(ctx, prompt...)
	log.Println("Finished processing image...", resp)
	if err != nil {
		log.Fatal(err)
		return "", err
	}

	return printResponse(resp), nil
}

// startNewChatSession	: Start a new chat session
func startNewChatSession() *genai.ChatSession {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(geminiKey))
	if err != nil {
		log.Fatal(err)
	}
	model := client.GenerativeModel("gemini-pro")
	value := float32(ChatTemperture)
	model.Temperature = &value
	cs := model.StartChat()
	return cs
}

// send: Send a message to the chat session
func send(cs *genai.ChatSession, msg string) *genai.GenerateContentResponse {
	if cs == nil {
		cs = startNewChatSession()
	}

	ctx := context.Background()
	log.Printf("== Me: %s\n== Model:\n", msg)
	res, err := cs.SendMessage(ctx, genai.Text(msg))
	if err != nil {
		log.Fatal(err)
	}
	return res
}

// Print response
func printResponse(resp *genai.GenerateContentResponse) string {
	var ret string
	for _, cand := range resp.Candidates {
		for _, part := range cand.Content.Parts {
			ret = ret + fmt.Sprintf("%v", part)
			log.Println(part)
		}
	}
	return ret
}
