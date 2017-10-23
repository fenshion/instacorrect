# Embed the characters
decoder_inputs = tf.Print(decoder_inputs, [decoder_inputs], message="decoder_inputs")
decoder_outputs = tf.Print(decoder_outputs, [decoder_outputs], message="decoder_outputs")
inputs = tf.nn.embedding_lookup(embeddings_w, decoder_inputs)
inputs = project_embedding(inputs, hidden_size)
# Get the positional embeddings
output_pos_embbed = encode_positions(position_table, inputs)
# # Add the positional embeddings
inputs += output_pos_embbed
output = decode_step(encoder_outputs, inputs, num_blocks,
                     dropout, num_heads, inference=True)
logits = make_logits(output, vocab_size_word)
preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
preds = tf.Print(preds, [preds], message="preds", summarize=10)
# Add these preds to the inputs
decoder_inputs = tf.concat([decoder_inputs, preds], axis=1)
decoder_outputs = tf.concat([decoder_outputs, output], axis=1)
i = tf.add(i, 1)
return i, decoder_inputs, decoder_outputs
