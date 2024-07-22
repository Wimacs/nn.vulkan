#include "vulkanexamplebase.h"

#define MAX_NEURONS_PER_LAYER 64
#define LAYER_COUNT 4

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float uv[2];
};

struct NNData
{
	uint32_t frameNumber;
	uint32_t outputWidth;
	uint32_t outputHeight;
	float learningRate;

	float rcpBatchSize;
	uint32_t batchSize;
	float adamBeta1;
	float adamBeta2;

	float adamEpsilon;
	float adamBeta1T;
	float adamBeta2T;
	uint32_t frequencies;

	uint32_t layerCount;
}g_data;

uint32_t divRoundUp(uint32_t x, uint32_t div)
{
	return (x + div - 1) / div;
}

class VulkanExample : public VulkanExampleBase
{
public:
	// Input image
	vks::Texture2D textureColorMap;
	// Storage image that the compute shader uses to apply the filter effect to
	vks::Texture2D storageImage;

	vks::Buffer weights;
	vks::Buffer Biases;

	vks::Buffer gradientWeights;
	vks::Buffer gradientBiases;

	vks::Buffer adamWeightsMeans;
	vks::Buffer adamWeightsVariances;
	vks::Buffer adamBiasesMeans;
	vks::Buffer adamBiasesVariances;

	vks::Buffer neuronsPerLayerDevice;
	vks::Buffer connectionDataBaseOffsetsDevice;
	vks::Buffer neuronDataBaseOffsetsDevice;

	vks::Buffer  nnuniformData;

	bool mNNNeedsInitialization = true;

	// Resources for the graphics part of the example
	struct Graphics {
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };	// Image display shader binding layout
		VkDescriptorSet descriptorSetPreCompute{ VK_NULL_HANDLE };		// Image display shader bindings before compute shader image manipulation
		VkDescriptorSet descriptorSetPostCompute{ VK_NULL_HANDLE };		// Image display shader bindings after compute shader image manipulation
		VkPipeline pipeline{ VK_NULL_HANDLE };							// Image display pipeline
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };				// Layout of the graphics pipeline
		VkSemaphore semaphore{ VK_NULL_HANDLE };						// Execution dependency between compute & graphic submission
		// Used to pass data to the graphics shaders
		struct UniformData {
			glm::mat4 projection;
			glm::mat4 modelView;
		} uniformData;
		vks::Buffer uniformBuffer;
	} graphics;

	// Resources for the compute part of the example
	struct Compute {
		VkQueue queue{ VK_NULL_HANDLE };								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool{ VK_NULL_HANDLE };					// Use a separate command pool (queue family may differ from the one used for graphics)
		VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };				// Command buffer storing the dispatch commands and barriers
		VkCommandBuffer initnnCommandBuffer{ VK_NULL_HANDLE };
		VkSemaphore semaphore{ VK_NULL_HANDLE };						// Execution dependency between compute & graphic submission
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };	// Compute shader binding layout
		VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };				// Compute shader bindings
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };				// Layout of the compute pipeline
		std::vector<VkPipeline> pipelines{};							// Compute pipelines for image filters
		int32_t pipelineIndex{ 0 };										// Current image filtering compute pipeline index
	} compute;

	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	uint32_t indexCount{ 0 };
	uint32_t vertexBufferSize{ 0 };

	std::vector<std::string> filterNames{};

	VulkanExample() : VulkanExampleBase()
	{
		title = "Compute shader image load/store";
		camera.type = Camera::CameraType::lookat;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -2.0f));
		camera.setRotation(glm::vec3(0.0f));
		camera.setPerspective(60.0f, (float)width * 0.5f / (float)height, 1.0f, 256.0f);
	}

	~VulkanExample()
	{
		if (device) {
			// Graphics
			vkDestroyPipeline(device, graphics.pipeline, nullptr);
			vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);
			vkDestroySemaphore(device, graphics.semaphore, nullptr);
			graphics.uniformBuffer.destroy();

			// Compute
			for (auto& pipeline : compute.pipelines)
			{
				vkDestroyPipeline(device, pipeline, nullptr);
			}
			vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
			vkDestroySemaphore(device, compute.semaphore, nullptr);
			vkDestroyCommandPool(device, compute.commandPool, nullptr);

			vertexBuffer.destroy();
			indexBuffer.destroy();

			textureColorMap.destroy();
			storageImage.destroy();
			nnuniformData.destroy();

			weights.destroy();
			Biases.destroy();
		}
	}

	// Prepare a storage image that is used to store the compute shader filter
	void prepareStorageImage()
	{
		const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

		VkFormatProperties formatProperties;
		// Get device properties for the requested texture format
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
		// Check if requested image format supports image storage operations required for storing pixel from the compute shader
		assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

		// Prepare blit target texture
		storageImage.width = textureColorMap.width;
		storageImage.height = textureColorMap.height;

		VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.extent = { storageImage.width, storageImage.height, 1 };
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		// Image will be sampled in the fragment shader and used as storage target in the compute shader
		imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		imageCreateInfo.flags = 0;
		// If compute and graphics queue family indices differ, we create an image that can be shared between them
		// This can result in worse performance than exclusive sharing mode, but save some synchronization to keep the sample simple
		std::vector<uint32_t> queueFamilyIndices;
		if (vulkanDevice->queueFamilyIndices.graphics != vulkanDevice->queueFamilyIndices.compute) {
			queueFamilyIndices = {
				vulkanDevice->queueFamilyIndices.graphics,
				vulkanDevice->queueFamilyIndices.compute
			};
			imageCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
			imageCreateInfo.queueFamilyIndexCount = 2;
			imageCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
		}
		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &storageImage.image));

		VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, storageImage.image, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &storageImage.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, storageImage.image, storageImage.deviceMemory, 0));

		// Transition image to the general layout, so we can use it as a storage image in the compute shader
		VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		storageImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		vks::tools::setImageLayout(layoutCmd, storageImage.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, storageImage.imageLayout);
		vulkanDevice->flushCommandBuffer(layoutCmd, queue, true);

		// Create sampler
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.compareOp = VK_COMPARE_OP_NEVER;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &storageImage.sampler));

		// Create image view
		VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.image = VK_NULL_HANDLE;
		view.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view.format = format;
		view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		view.image = storageImage.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &storageImage.view));

		// Initialize a descriptor for later use
		storageImage.descriptor.imageLayout = storageImage.imageLayout;
		storageImage.descriptor.imageView = storageImage.view;
		storageImage.descriptor.sampler = storageImage.sampler;
		storageImage.device = vulkanDevice;
	}

	void prepareBuffers()
	{
		//create weights buffer

		int mMaxWeightsCount = (LAYER_COUNT - 1) * MAX_NEURONS_PER_LAYER * MAX_NEURONS_PER_LAYER;
		//std::vector<float> a, b;
		//for (int i = 0; i < mMaxWeightsCount; i++)
		//	a.push_back(0.0f);


		VkDeviceSize weightsBufferSize = mMaxWeightsCount * sizeof(float);



		//vks::Buffer stagingBuffer;

		//vulkanDevice->createBuffer(
		//	VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		//	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		//	&stagingBuffer,
		//	weightsBufferSize,
		//	a.data());

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&weights,
			weightsBufferSize);

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT ,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&gradientWeights,
			weightsBufferSize);

		//// Copy from staging buffer
		//VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		//VkBufferCopy copyRegion = {};
		//copyRegion.size = mMaxWeightsCount;
		//vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, weights.buffer, 1, &copyRegion);
		//vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, gradientWeights.buffer, 1, &copyRegion);

		//// Add an initial release barrier to the graphics queue,
		//// so that when the compute command buffer executes for the first time
		//// it doesn't complain about a lack of a corresponding "release" to its "acquire"
		////addGraphicsToComputeBarriers(copyCmd, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 0, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
		//vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		//stagingBuffer.destroy();

		//crate biases buffer
		int mMaxBiasesCount = (LAYER_COUNT - 1) * MAX_NEURONS_PER_LAYER;
		VkDeviceSize biasesBufferSize = mMaxBiasesCount * sizeof(float);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&Biases,
			biasesBufferSize);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&gradientBiases,
			biasesBufferSize);

		// Create buffers for Adam optimizer
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&adamWeightsMeans,
			weightsBufferSize);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&adamWeightsVariances,
			weightsBufferSize);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&adamBiasesMeans,
			biasesBufferSize);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&adamBiasesVariances,
			biasesBufferSize);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&nnuniformData,
			sizeof(NNData)));

		VK_CHECK_RESULT(nnuniformData.map());
		updateUniformBuffer();

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&neuronsPerLayerDevice,
			LAYER_COUNT * sizeof(float));

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&connectionDataBaseOffsetsDevice,
			LAYER_COUNT * sizeof(float));

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&neuronDataBaseOffsetsDevice,
			LAYER_COUNT * sizeof(float));

		updateNNArch();
	}

	void updateNNArch()
	{

		int neuronsPerLayer[LAYER_COUNT];
		// Figure out number of neurons per layer
		{
			neuronsPerLayer[0] = 2 /*getInputLayerNeuronCount()*/;
			neuronsPerLayer[LAYER_COUNT - 1] = 3; //RGB output
			for (int i = 1; i < LAYER_COUNT - 1; i++)
			{
				neuronsPerLayer[i] = MAX_NEURONS_PER_LAYER;
			}
		}
		{
			vks::Buffer stagingBuffer;
			std::vector<uint32_t> neuronsPerLayerHost;
			for (int i = 0; i < LAYER_COUNT; i++)
			{
				neuronsPerLayerHost.push_back(neuronsPerLayer[i]);
			}
			vulkanDevice->createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				&stagingBuffer,
				LAYER_COUNT * sizeof(float),
				neuronsPerLayerHost.data());

			// Copy from staging buffer
			VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
			VkBufferCopy copyRegion = {};
			copyRegion.size = LAYER_COUNT * sizeof(float);
			vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, neuronsPerLayerDevice.buffer, 1, &copyRegion);
			vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
			stagingBuffer.destroy();
		}
		{
			glm::uint offset = 0;
			vks::Buffer stagingBuffer;
			std::vector<uint32_t> connectionDataBaseOffsetsHost;
			connectionDataBaseOffsetsHost.push_back(0);
			for (int i = 1; i < LAYER_COUNT; i++)
			{
				connectionDataBaseOffsetsHost.push_back(offset);
				offset += neuronsPerLayer[i - 1] * neuronsPerLayer[i];
			}
			vulkanDevice->createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				&stagingBuffer,
				LAYER_COUNT * sizeof(float),
				connectionDataBaseOffsetsHost.data());

			// Copy from staging buffer
			VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
			VkBufferCopy copyRegion = {};
			copyRegion.size = LAYER_COUNT * sizeof(float);
			vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, connectionDataBaseOffsetsDevice.buffer, 1, &copyRegion);
			vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
			stagingBuffer.destroy();
		}
		{
			glm::uint offset = 0;
			vks::Buffer stagingBuffer;
			std::vector<uint32_t> neuronDataBaseOffsetsHost;
			for (int i = 0; i < LAYER_COUNT; i++)
			{
				neuronDataBaseOffsetsHost.push_back(offset);
				offset += neuronsPerLayer[i];
			}
			vulkanDevice->createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				&stagingBuffer,
				LAYER_COUNT * sizeof(float),
				neuronDataBaseOffsetsHost.data());

			// Copy from staging buffer
			VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
			VkBufferCopy copyRegion = {};
			copyRegion.size = LAYER_COUNT * sizeof(float);
			vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, neuronDataBaseOffsetsDevice.buffer, 1, &copyRegion);
			vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
			stagingBuffer.destroy();
		}
	}

	void updateUniformBuffer()
	{
		g_data.frameNumber = frameCounter;
		g_data.layerCount = LAYER_COUNT;
		g_data.outputHeight = textureColorMap.height;
		g_data.outputWidth = textureColorMap.width;
		g_data.batchSize = 4096;
		g_data.rcpBatchSize = 1.0f / g_data.batchSize;
		g_data.learningRate = 0.01f;

		memcpy(nnuniformData.mapped, &g_data, sizeof(NNData));
	}

	void loadAssets()
	{
		textureColorMap.loadFromFile(getAssetPath() + "textures/vulkan_11_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_LAYOUT_GENERAL);
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// Image memory barrier to make sure that compute shader writes are finished before sampling from the texture
			VkImageMemoryBarrier imageMemoryBarrier = {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			// We won't be changing the layout of the image
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = storageImage.image;
			imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(
				drawCmdBuffers[i],
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width * 0.5f, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertexBuffer.buffer, offsets);
			vkCmdBindIndexBuffer(drawCmdBuffers[i], indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			// Left (pre compute)
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSetPreCompute, 0, NULL);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);

			vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);

			// Right (post compute)
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSetPostCompute, 0, NULL);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);

			viewport.x = (float)width / 2.0f;
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
			vkCmdDrawIndexed(drawCmdBuffers[i], indexCount, 1, 0, 0, 0);

			drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}

	}

	void buildComputeCommandBuffer()
	{


		//nn init
		//if (mNNNeedsInitialization)
		{
			// Flush the queue if we're rebuilding the command buffer after a pipeline change to ensure it's not currently in use
			vkQueueWaitIdle(compute.queue);

			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

			VK_CHECK_RESULT(vkBeginCommandBuffer(compute.initnnCommandBuffer, &cmdBufInfo));

			mNNNeedsInitialization = false;
			vkCmdBindPipeline(compute.initnnCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[compute.pipelineIndex]);
			vkCmdBindDescriptorSets(compute.initnnCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

			const uint32_t dispatchWidth = divRoundUp(LAYER_COUNT, 16);
			const uint32_t dispatchHeight = divRoundUp(MAX_NEURONS_PER_LAYER, 16);

			vkCmdDispatch(compute.initnnCommandBuffer, dispatchWidth, dispatchHeight, 1);

			vkEndCommandBuffer(compute.initnnCommandBuffer);
		}



		//nn traning + inference
		{
			vkQueueWaitIdle(compute.queue);

			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

			{
				vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[compute.pipelineIndex + 2]);
				vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

				const uint32_t dispatchWidth = divRoundUp(LAYER_COUNT, 16);
				const uint32_t dispatchHeight = divRoundUp(MAX_NEURONS_PER_LAYER, 16);

				vkCmdDispatch(compute.commandBuffer, dispatchWidth, dispatchHeight, 1);
			}

			{
				vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[compute.pipelineIndex + 3]);
				//vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

				const uint32_t dispatchWidth = divRoundUp(g_data.batchSize, 8);
				vkCmdDispatch(compute.commandBuffer, dispatchWidth, 1, 1);
			}

			{
				vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[compute.pipelineIndex + 4]);
				//vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

				const uint32_t dispatchWidth = divRoundUp(LAYER_COUNT, 16);
				const uint32_t dispatchHeight = divRoundUp(MAX_NEURONS_PER_LAYER, 16);

				vkCmdDispatch(compute.commandBuffer, dispatchWidth, dispatchHeight, 1);
			}

			{
				vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[compute.pipelineIndex + 1]);
				//vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

				const uint32_t dispatchWidth = divRoundUp(textureColorMap.width, 8);
				const uint32_t dispatchHeight = divRoundUp(textureColorMap.height, 8);

				vkCmdDispatch(compute.commandBuffer, dispatchWidth, dispatchHeight, 1);
			}


			vkEndCommandBuffer(compute.commandBuffer);

		}


	}

	// Setup vertices for a single uv-mapped quad used to display the input and output images
	void generateQuad()
	{
		// Setup vertices for a single uv-mapped quad made from two triangles
		std::vector<Vertex> vertices = {
			{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f } },
			{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f } },
			{ { -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
			{ {  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } }
		};

		// Setup indices
		std::vector<uint32_t> indices = { 0,1,2, 2,3,0 };
		indexCount = static_cast<uint32_t>(indices.size());

		// Create buffers and upload data to the GPU

		struct StagingBuffers {
			vks::Buffer vertices;
			vks::Buffer indices;
		} stagingBuffers;

		// Host visible source buffers (staging)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffers.vertices, vertices.size() * sizeof(Vertex), vertices.data()));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffers.indices, indices.size() * sizeof(uint32_t), indices.data()));

		// Device local destination buffers
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &vertexBuffer, vertices.size() * sizeof(Vertex)));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &indexBuffer, indices.size() * sizeof(uint32_t)));

		// Copy from host do device
		vulkanDevice->copyBuffer(&stagingBuffers.vertices, &vertexBuffer, queue);
		vulkanDevice->copyBuffer(&stagingBuffers.indices, &indexBuffer, queue);

		// Clean up
		stagingBuffers.vertices.destroy();
		stagingBuffers.indices.destroy();
	}

	// The descriptor pool will be shared between graphics and compute
	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			// Graphics pipelines uniform buffers
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6),
			// Graphics pipelines image samplers for displaying compute output image
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2),
			// Compute pipelines uses a storage image for image reads and writes
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2),
			// Compute pipllines
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8),
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 4);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	// Prepare the graphics resources used to display the ray traced output of the compute shader
	void prepareGraphics()
	{
		// Create a semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &graphics.semaphore));

		// Signal the semaphore
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &graphics.semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VK_CHECK_RESULT(vkQueueWaitIdle(queue));

		// Setup descriptors

		// The graphics pipeline uses two sets with two bindings
		// One set for displaying the input image and one set for displaying the output image with the compute filter applied
		// Binding 0: Vertex shader uniform buffer
		// Binding 1: Sampled image (before/after compute filter is applied)

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphics.descriptorSetLayout, 1);

		// Input image (before compute post processing)
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSetPreCompute));
		std::vector<VkWriteDescriptorSet> baseImageWriteDescriptorSets = {
			vks::initializers::writeDescriptorSet(graphics.descriptorSetPreCompute, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &graphics.uniformBuffer.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSetPreCompute, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textureColorMap.descriptor)
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(baseImageWriteDescriptorSets.size()), baseImageWriteDescriptorSets.data(), 0, nullptr);

		// Final image (after compute shader processing)
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSetPostCompute));
		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(graphics.descriptorSetPostCompute, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &graphics.uniformBuffer.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSetPostCompute, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &storageImage.descriptor)
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		// Graphics pipeline used to display the images (before and after the compute effect is applied)

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		// Shaders
		shaderStages[0] = loadShader(getShadersPath() + "mlp/texture.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "mlp/texture.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		// Vertex input state
		std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
		};
		std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass, 0);
		pipelineCreateInfo.pVertexInputState = &vertexInputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));
	}

	void prepareCompute()
	{
		// Get a compute queue from the device
		vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Input image (read-only)
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			// Binding 1: Output image (write)
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1),

			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),

			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),

			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 8),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 9),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 10),

			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 11),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 12),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 13),

		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device,	&descriptorLayout, nullptr, &compute.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0, &textureColorMap.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImage.descriptor),

			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &weights.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &Biases.descriptor),

			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &gradientBiases.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &gradientWeights.descriptor),

			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6, &adamBiasesMeans.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7, &adamBiasesVariances.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8, &adamWeightsMeans.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9, &adamWeightsVariances.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10, &nnuniformData.descriptor),

			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11, &neuronsPerLayerDevice.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12, &connectionDataBaseOffsetsDevice.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13, &neuronDataBaseOffsetsDevice.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);

		// Create compute shader pipelines
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

		// One pipeline for each available image filter
		filterNames = { "nninit", "nninference", "nncleargradients", "nntraining", "nnoptimize"};
		for (auto& shaderName : filterNames) {
			std::string fileName = getShadersPath() + "mlp/" + shaderName + ".comp.spv";
			computePipelineCreateInfo.stage = loadShader(fileName, VK_SHADER_STAGE_COMPUTE_BIT);
			VkPipeline pipeline;
			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));
			compute.pipelines.push_back(pipeline);
		}

		// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo( compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.initnnCommandBuffer));


		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &graphics.uniformBuffer, sizeof(Graphics::UniformData)));
		// Map persistent
		VK_CHECK_RESULT(graphics.uniformBuffer.map());
	}

	void updateUniformBuffers()
	{
		// We need to adjust the perspective as this sample displays two viewports side-by-side
		camera.setPerspective(60.0f, (float)width * 0.5f / (float)height, 1.0f, 256.0f);
		graphics.uniformData.projection = camera.matrices.perspective;
		graphics.uniformData.modelView = camera.matrices.view;
		memcpy(graphics.uniformBuffer.mapped, &graphics.uniformData, sizeof(Graphics::UniformData));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		generateQuad();
		prepareUniformBuffers();
		prepareStorageImage();
		prepareBuffers();
		setupDescriptorPool();
		prepareCompute();
		prepareGraphics();
		buildCommandBuffers();
		prepared = true;
	}

	void draw()
	{
		// Wait for rendering finished
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		// Submit compute commands
		if (mNNNeedsInitialization)
		{
			mNNNeedsInitialization = false;
			VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &compute.initnnCommandBuffer;
			computeSubmitInfo.waitSemaphoreCount = 1;
			computeSubmitInfo.pWaitSemaphores = &graphics.semaphore;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.signalSemaphoreCount = 1;
			computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
			VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
		}

		{
			VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
			computeSubmitInfo.waitSemaphoreCount = 1;
			computeSubmitInfo.pWaitSemaphores = &graphics.semaphore;
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.signalSemaphoreCount = 1;
			computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
			VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
		}
		VulkanExampleBase::prepareFrame();

		VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, semaphores.presentComplete };
		VkSemaphore graphicsSignalSemaphores[] = { graphics.semaphore, semaphores.renderComplete };

		// Submit graphics commands
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		submitInfo.waitSemaphoreCount = 2;
		submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
		submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	virtual void render()
	{
		if (!prepared)
		{
			return;
		}
		updateUniformBuffers();
		updateUniformBuffer();
		draw();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (ImGui::Button("Initialize Weights"))
		{
			mNNNeedsInitialization = true;
		}
	}
};

VULKAN_EXAMPLE_MAIN()
