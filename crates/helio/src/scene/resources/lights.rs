//! Light resource management for the scene.
//!
//! Lights are stored in a dense arena and uploaded to GPU storage buffers.
//! Unlike other resources, lights have no reference counting (they exist
//! independently of objects).

use helio_v3::GpuLight;

use crate::handles::LightId;

use super::super::errors::{invalid, Result};
use super::super::types::LightRecord;

impl super::super::Scene {
    /// Insert a light into the scene.
    ///
    /// Adds the light to the dense arena and uploads it to the GPU light storage buffer.
    ///
    /// # Parameters
    /// - `light`: GPU light parameters:
    ///   - Position (for point/spot lights)
    ///   - Direction (for directional/spot lights)
    ///   - Color and intensity
    ///   - Light type (point, directional, spot)
    ///   - Shadow settings (shadow_index, shadow resolution)
    ///
    /// # Returns
    /// A [`LightId`] handle that can be used to update or remove the light.
    ///
    /// # Performance
    /// - CPU cost: O(1) insertion into dense arena
    /// - GPU cost: Pushes light data to GPU storage buffer
    /// - Memory: Lights are stored in a dense GPU storage buffer
    ///
    /// # Shadow Casting Limits
    /// The scene supports up to 42 shadow-casting lights (42 × 6 = 252 shadow atlas layers).
    /// Additional shadow-casting lights will have shadows disabled automatically.
    ///
    /// # Example
    /// ```ignore
    /// let light_id = scene.insert_light(GpuLight {
    ///     position: [0.0, 5.0, 0.0],
    ///     color: [1.0, 1.0, 1.0],
    ///     intensity: 100.0,
    ///     light_type: LightType::Point as u32,
    ///     shadow_index: 0, // Enable shadows (assigned automatically in flush())
    ///     ..Default::default()
    /// });
    /// ```
    pub(in crate::scene) fn insert_light(&mut self, light: GpuLight) -> LightId {
        self.insert_light_with_movability(light, None, 0)
    }

    /// Insert a light into the scene with explicit movability and user tag.
    pub(in crate::scene) fn insert_light_with_movability(
        &mut self,
        light: GpuLight,
        movability: Option<libhelio::Movability>,
        user_tag: u64,
    ) -> LightId {
        // Default lights to Movable (most common case for real-time lighting).
        // Static lights are opt-in for baking scenarios.
        let movability = movability.unwrap_or(libhelio::Movability::Movable);
        let (id, dense_index) = self.lights.insert(LightRecord {
            gpu: light,
            movability,
            user_tag,
        });
        let pushed = self.gpu_scene.lights.push(light);
        debug_assert_eq!(pushed, dense_index);

        // The movable set changed → flush() must rebuild the compacted runtime
        // lights buffer and re-assign shadow-caster slots.
        self.light_set_dirty = true;

        // Invalidate any previous bake if this is a static/stationary light
        if !movability.can_move() {
            self.bake_invalidated = true;
        }

        id
    }

    /// Update a light's parameters.
    ///
    /// Modifies the light's GPU parameters and updates the GPU storage buffer.
    ///
    /// # Parameters
    /// - `id`: Light handle
    /// - `light`: New GPU light parameters
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the light ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the light was successfully updated.
    ///
    /// # Performance
    /// - CPU cost: O(1)
    /// - GPU cost: Updates light storage buffer slot
    ///
    /// # Example
    /// ```ignore
    /// // Animate light intensity
    /// let mut light = scene.get_light(light_id)?;
    /// light.intensity = 200.0; // Brighten
    /// scene.update_light(light_id, light)?;
    /// ```
    pub fn update_light(&mut self, id: LightId, light: GpuLight) -> Result<()> {
        // Update the arena record and capture the Copy values we need, ending the
        // `&mut self.lights` borrow before touching `self.gpu_scene` / dirty flags.
        let (dense_index, old) = {
            let Some((dense_index, record)) = self.lights.get_mut_with_index(id) else {
                return Err(invalid("light"));
            };
            let old = record.gpu;

            if !record.movability.can_move() {
                // Static lights cannot move; reject position/direction edits.
                // Other edits update only the stored record — static lights are
                // baked and never enter the runtime (movable) lights buffer, so
                // there is nothing to upload and no shadow-budget impact.
                let position_changed = old.position_range != light.position_range;
                let direction_changed = old.direction_outer != light.direction_outer;
                if position_changed || direction_changed {
                    log::warn!(
                        "Attempted to update position/direction on Static light {:?}. Set movability to Movable to allow updates.",
                        id
                    );
                    return Ok(()); // No-op instead of error
                }
                record.gpu = light;
                return Ok(());
            }

            record.gpu = light;
            (dense_index, old)
        };

        // Movable light data changed → bump the generation counter so shadow
        // caching (per-caster hashes) and any lights_gen-keyed pass re-evaluate.
        self.movable_lights_generation = self.movable_lights_generation.wrapping_add(1);
        self.gpu_scene.movable_lights_generation = self.movable_lights_generation;

        // A change to a score-relevant field (importance = intensity × range²) or
        // to the shadow-enable flag changes the 42-caster budget, so force a full
        // rebuild + re-score on the next flush.  A pure position/direction drag
        // changes neither, and takes the in-place fast path below.
        let score_changed = old.color_intensity[3] != light.color_intensity[3]
            || old.position_range[3] != light.position_range[3];
        let shadow_enable_changed =
            (old.shadow_index == u32::MAX) != (light.shadow_index == u32::MAX);
        if score_changed || shadow_enable_changed {
            self.light_set_dirty = true;
            return Ok(());
        }

        // A full rebuild is already pending — flush() will read the fresh arena
        // data, so skip the incremental write (movable_slot_of may be stale).
        if self.light_set_dirty {
            return Ok(());
        }

        // Fast path: position/direction-only change.  Patch the single compacted
        // slot in place, preserving the GPU-assigned `shadow_index` (the arena /
        // caller copy carries the *request* value, not the assigned atlas slot).
        if let Some(&slot) = self.movable_slot_of.get(dense_index) {
            let slot = slot as usize;
            if slot < self.gpu_scene.lights.len() {
                let assigned_shadow = self.gpu_scene.lights.0.as_slice()[slot].shadow_index;
                let mut g = light;
                g.shadow_index = assigned_shadow;
                let updated = self.gpu_scene.lights.update(slot, g);
                debug_assert!(updated);
                return Ok(());
            }
        }

        // Mapping missing/stale (e.g. before the first flush) → force a rebuild.
        self.light_set_dirty = true;
        Ok(())
    }

    /// Remove a light from the scene.
    ///
    /// Removes the light from the dense arena and GPU storage buffer using swap-remove
    /// (the last light is moved to the removed light's slot for O(1) removal).
    ///
    /// # Parameters
    /// - `id`: Light handle
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the light ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the light was successfully removed.
    ///
    /// # Performance
    /// - CPU cost: O(1) swap-remove from dense arena
    /// - GPU cost: Swap-removes light from GPU storage buffer
    ///
    /// # Example
    /// ```ignore
    /// scene.remove_light(light_id)?;
    /// ```
    pub fn remove_light(&mut self, id: LightId) -> Result<()> {
        let removed = self.lights.remove(id).ok_or_else(|| invalid("light"))?;
        let gpu_removed = self.gpu_scene.lights.swap_remove(removed.dense_index);
        debug_assert!(gpu_removed.is_some());

        // The movable set changed → flush() must rebuild the compacted runtime
        // lights buffer and re-assign shadow-caster slots.
        self.light_set_dirty = true;

        Ok(())
    }
}

